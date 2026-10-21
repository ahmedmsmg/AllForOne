//! Encodings for hidden states crossing a pipeline boundary.
//!
//! Every stage boundary goes through an [`ActivationCodec`], so compression (and later, privacy
//! schemes such as residue-number-system sharing) can change without touching the transport or
//! the engine. Hidden states are row-major `(n_tokens, d)` f32 buffers; scaled codecs keep one
//! f32 scale per token row, because Llama's residual stream has large per-token outliers.

use std::{fmt, str::FromStr};

use anyhow::{Result, bail, ensure};
use float8::F8E4M3;
use half::f16;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecKind {
    F32 = 0,
    F16 = 1,
    Fp8 = 2,
    Int8 = 3,
}

impl CodecKind {
    pub const ALL: [CodecKind; 4] = [Self::F32, Self::F16, Self::Fp8, Self::Int8];

    pub fn from_u8(v: u8) -> Result<Self> {
        match Self::ALL.get(v as usize) {
            Some(k) => Ok(*k),
            None => bail!("unknown codec id {v}"),
        }
    }

    pub fn codec(self) -> &'static dyn ActivationCodec {
        match self {
            Self::F32 => &F32Codec,
            Self::F16 => &F16Codec,
            Self::Fp8 => &Fp8Codec,
            Self::Int8 => &Int8Codec,
        }
    }
}

impl FromStr for CodecKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" => Self::F32,
            "f16" => Self::F16,
            "fp8" => Self::Fp8,
            "int8" => Self::Int8,
            _ => bail!("unknown codec {s:?} (f32 | f16 | fp8 | int8)"),
        })
    }
}

impl fmt::Display for CodecKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Fp8 => "fp8",
            Self::Int8 => "int8",
        })
    }
}

pub trait ActivationCodec: Sync {
    fn kind(&self) -> CodecKind;
    /// Appends the encoding of `x` (rows of width `d`) to `out`.
    fn encode(&self, x: &[f32], d: usize, out: &mut Vec<u8>);
    /// Decodes `bytes` holding `n_tokens` rows of width `d`.
    fn decode(&self, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>>;
}

pub fn encode(kind: CodecKind, x: &[f32], d: usize) -> Vec<u8> {
    let mut out = Vec::new();
    kind.codec().encode(x, d, &mut out);
    out
}

pub fn decode(kind: CodecKind, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>> {
    kind.codec().decode(bytes, n_tokens, d)
}

fn check_len(bytes: &[u8], expected: usize) -> Result<()> {
    ensure!(bytes.len() == expected, "payload is {} bytes, expected {expected}", bytes.len());
    Ok(())
}

struct F32Codec;

impl ActivationCodec for F32Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::F32
    }
    fn encode(&self, x: &[f32], _d: usize, out: &mut Vec<u8>) {
        out.reserve(x.len() * 4);
        out.extend(x.iter().flat_map(|v| v.to_le_bytes()));
    }
    fn decode(&self, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>> {
        check_len(bytes, n_tokens * d * 4)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

struct F16Codec;

impl ActivationCodec for F16Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::F16
    }
    fn encode(&self, x: &[f32], _d: usize, out: &mut Vec<u8>) {
        out.reserve(x.len() * 2);
        out.extend(x.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()));
    }
    fn decode(&self, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>> {
        check_len(bytes, n_tokens * d * 2)?;
        Ok(bytes.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
    }
}

/// Shared layout for the 8-bit codecs: per row, an f32 scale followed by `d` one-byte values.
fn encode_scaled(x: &[f32], d: usize, max_code: f32, out: &mut Vec<u8>, quant: impl Fn(f32) -> u8) {
    out.reserve(x.len() + x.len() / d * 4);
    for row in x.chunks_exact(d) {
        let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if absmax > 0. { absmax / max_code } else { 1. };
        out.extend(scale.to_le_bytes());
        out.extend(row.iter().map(|v| quant(v / scale)));
    }
}

fn decode_scaled(bytes: &[u8], n_tokens: usize, d: usize, dequant: impl Fn(u8) -> f32) -> Result<Vec<f32>> {
    check_len(bytes, n_tokens * (d + 4))?;
    let mut out = Vec::with_capacity(n_tokens * d);
    for row in bytes.chunks_exact(d + 4) {
        let scale = f32::from_le_bytes(row[..4].try_into().unwrap());
        out.extend(row[4..].iter().map(|b| dequant(*b) * scale));
    }
    Ok(out)
}

struct Fp8Codec;

impl ActivationCodec for Fp8Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::Fp8
    }
    fn encode(&self, x: &[f32], d: usize, out: &mut Vec<u8>) {
        encode_scaled(x, d, 448., out, |v| F8E4M3::from_f32(v).to_bits());
    }
    fn decode(&self, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>> {
        decode_scaled(bytes, n_tokens, d, |b| F8E4M3::from_bits(b).to_f32())
    }
}

struct Int8Codec;

impl ActivationCodec for Int8Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::Int8
    }
    fn encode(&self, x: &[f32], d: usize, out: &mut Vec<u8>) {
        encode_scaled(x, d, 127., out, |v| v.round().clamp(-127., 127.) as i8 as u8);
    }
    fn decode(&self, bytes: &[u8], n_tokens: usize, d: usize) -> Result<Vec<f32>> {
        decode_scaled(bytes, n_tokens, d, |b| b as i8 as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_bounds() {
        let d = 64;
        // Two rows, the second with an outlier like Llama's residual stream has.
        let mut x: Vec<f32> = (0..2 * d).map(|i| ((i * 37 % 101) as f32 - 50.) / 10.).collect();
        x[d + 3] = 900.;
        for (kind, tol) in [(CodecKind::F32, 0.), (CodecKind::F16, 1e-3), (CodecKind::Fp8, 0.07), (CodecKind::Int8, 0.004)] {
            let y = decode(kind, &encode(kind, &x, d), 2, d).unwrap();
            for (row_x, row_y) in x.chunks(d).zip(y.chunks(d)) {
                let absmax = row_x.iter().fold(0f32, |m, v| m.max(v.abs()));
                for (a, b) in row_x.iter().zip(row_y) {
                    assert!((a - b).abs() <= tol * absmax.max(a.abs()), "{kind}: {a} -> {b}");
                }
            }
        }
    }

    #[test]
    fn rejects_bad_length() {
        assert!(decode(CodecKind::F16, &[0; 5], 1, 4).is_err());
        assert!(CodecKind::from_u8(9).is_err());
    }
}

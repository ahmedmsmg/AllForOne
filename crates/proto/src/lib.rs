//! Wire messages and the QUIC transport shared by nodes and clients.
//!
//! Data moves around a ring: client -> node 1 -> ... -> node N -> client. Each hop sends straight
//! to its successor (push-forward), so a token costs N one-way hops rather than N round trips.

pub mod http;
pub mod status;
mod transport;

use std::net::SocketAddr;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use transport::{Inbound, Net, NetOpts};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg {
    /// Sets up `session` along `route`. Each hop records `route[hop + 1]` as its successor and
    /// passes the message on; the final entry is the client, which treats arrival as the ack.
    Open { session: u64, route: Vec<SocketAddr>, hop: u32 },
    Forward(Forward),
    /// Tears the session down along the ring (also returns to the client as an ack).
    Close { session: u64 },
    /// Sent directly to the client when a hop fails.
    Error { session: u64, hop: u32, message: String },
}

/// Hidden states for `n_tokens` tokens at positions `pos..pos + n_tokens`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forward {
    pub session: u64,
    /// Client-chosen id that comes back unchanged, to match replies with requests.
    pub step: u64,
    pub pos: u32,
    pub n_tokens: u32,
    pub codec: u8,
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
    /// Appended by every hop so the client can split latency into compute and network time.
    pub hops: Vec<HopStat>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HopStat {
    /// Time from receiving the frame to handing the result to the transport (decode + layers +
    /// encode), in microseconds.
    pub busy_us: u32,
}

impl Msg {
    pub fn to_frame(&self) -> Result<Vec<u8>> {
        // 4-byte little-endian length prefix, then the postcard body.
        let mut frame = postcard::to_extend(self, vec![0u8; 4])?;
        let len = (frame.len() - 4) as u32;
        frame[..4].copy_from_slice(&len.to_le_bytes());
        Ok(frame)
    }

    pub fn from_body(body: &[u8]) -> Result<Self> {
        Ok(postcard::from_bytes(body)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() {
        let msg = Msg::Forward(Forward {
            session: 7,
            step: 3,
            pos: 12,
            n_tokens: 1,
            codec: 1,
            payload: vec![1, 2, 3, 250],
            hops: vec![HopStat { busy_us: 99 }],
        });
        let frame = msg.to_frame().unwrap();
        let len = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        assert_eq!(len, frame.len() - 4);
        match Msg::from_body(&frame[4..]).unwrap() {
            Msg::Forward(f) => {
                assert_eq!((f.session, f.step, f.pos, f.payload), (7, 3, 12, vec![1, 2, 3, 250]));
                assert_eq!(f.hops[0].busy_us, 99);
            }
            m => panic!("wrong message {m:?}"),
        }
    }
}

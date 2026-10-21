//! Group planning: each machine runs [`probe`] to report what it can offer, and [`plan`] picks
//! the best model from the [`catalog`] that the pooled hardware can run, and how to split it.

pub mod catalog;
pub mod plan;
pub mod probe;

/// Bytes as decimal gigabytes, the unit model downloads are usually quoted in.
pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

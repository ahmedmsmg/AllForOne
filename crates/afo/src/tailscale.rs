//! What this machine's Tailscale knows: its own address and the other machines on the tailnet.

use std::{net::IpAddr, process::Command};

use anyhow::{Context, Result, bail};
use serde_json::Value;

pub struct Tailnet {
    pub me: IpAddr,
    /// Addresses of the other machines that are online right now.
    pub peers: Vec<IpAddr>,
}

pub fn status() -> Result<Tailnet> {
    let out = Command::new("tailscale")
        .args(["status", "--json"])
        .output()
        .context("running `tailscale` (is Tailscale installed? the installer sets it up)")?;
    if !out.status.success() {
        bail!("tailscale status failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let v: Value = serde_json::from_slice(&out.stdout).context("parsing `tailscale status --json`")?;
    if v["BackendState"].as_str() != Some("Running") {
        bail!("Tailscale is not connected (state {}); run `tailscale up`", v["BackendState"]);
    }
    let first_ipv4 = |node: &Value| {
        node["TailscaleIPs"].as_array()?.iter().filter_map(|ip| ip.as_str()?.parse::<IpAddr>().ok()).find(IpAddr::is_ipv4)
    };
    let me = first_ipv4(&v["Self"]).context("this machine has no Tailscale IPv4 address")?;
    let peers = v["Peer"]
        .as_object()
        .map(|peers| {
            peers
                .values()
                .filter(|p| p["Online"].as_bool() == Some(true))
                .filter_map(first_ipv4)
                .collect()
        })
        .unwrap_or_default();
    Ok(Tailnet { me, peers })
}

/// Tailscale hands out addresses from 100.64.0.0/10 and fd7a:115c:a1e0::/48.
pub fn is_tailnet(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            a == 100 && (64..128).contains(&b)
        }
        IpAddr::V6(v6) => v6.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tailnet_ranges() {
        let yes = ["100.64.0.1", "100.127.255.254", "fd7a:115c:a1e0::1"];
        let no = ["100.63.255.255", "100.128.0.1", "192.168.1.5", "127.0.0.1", "fd7a:115c:a1e1::1"];
        assert!(yes.iter().all(|ip| is_tailnet(ip.parse().unwrap())));
        assert!(no.iter().all(|ip| !is_tailnet(ip.parse().unwrap())));
    }
}

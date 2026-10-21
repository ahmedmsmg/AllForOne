//! Just enough HTTP/1.1 for status pages: GET requests, one per connection, small bodies.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::timeout,
};
use tracing::debug;

const MAX_REQUEST: usize = 16 << 10;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(body: String) -> Self {
        Self { status: 200, content_type: "application/json", body: body.into_bytes() }
    }

    pub fn html(body: &str) -> Self {
        Self { status: 200, content_type: "text/html; charset=utf-8", body: body.as_bytes().to_vec() }
    }

    pub fn not_found() -> Self {
        Self { status: 404, content_type: "text/plain", body: b"not found\n".to_vec() }
    }
}

/// Answers GET requests on `listener` with `handler(path)` until the task is dropped.
pub async fn serve(listener: TcpListener, handler: impl Fn(&str) -> Response + Send + Sync + 'static) {
    let handler = Arc::new(handler);
    loop {
        let Ok((sock, peer)) = listener.accept().await else { continue };
        let handler = handler.clone();
        tokio::spawn(async move {
            if let Err(e) = respond(sock, &*handler).await {
                debug!("http request from {peer}: {e:#}");
            }
        });
    }
}

async fn respond(mut sock: TcpStream, handler: &(impl Fn(&str) -> Response + ?Sized)) -> Result<()> {
    let mut req = Vec::with_capacity(1024);
    timeout(IO_TIMEOUT, async {
        while !req.windows(4).any(|w| w == b"\r\n\r\n") {
            ensure!(req.len() < MAX_REQUEST, "request headers too large");
            ensure!(sock.read_buf(&mut req).await? > 0, "connection closed mid-request");
        }
        Ok(())
    })
    .await
    .context("timed out reading request")??;

    let line = String::from_utf8_lossy(&req[..req.iter().position(|&b| b == b'\r').unwrap_or(0)]).into_owned();
    let mut parts = line.split(' ');
    let resp = match (parts.next(), parts.next()) {
        (Some("GET"), Some(target)) => handler(target.split('?').next().unwrap_or(target)),
        _ => Response { status: 405, content_type: "text/plain", body: b"only GET\n".to_vec() },
    };
    let reason = match resp.status {
        200 => "OK",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "",
    };
    let head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        resp.status,
        resp.content_type,
        resp.body.len()
    );
    timeout(IO_TIMEOUT, async {
        sock.write_all(head.as_bytes()).await?;
        sock.write_all(&resp.body).await?;
        sock.shutdown().await
    })
    .await
    .context("timed out writing response")??;
    Ok(())
}

/// Fetches `path` from `addr` and returns the body of a 200 response.
pub async fn get(addr: SocketAddr, path: &str, limit: Duration) -> Result<Vec<u8>> {
    timeout(limit, async {
        let mut sock = TcpStream::connect(addr).await?;
        sock.write_all(format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes()).await?;
        let mut resp = Vec::new();
        sock.read_to_end(&mut resp).await?;
        let split = resp.windows(4).position(|w| w == b"\r\n\r\n").context("malformed HTTP response")?;
        let status = String::from_utf8_lossy(&resp[..split]).lines().next().unwrap_or_default().to_string();
        if status.split(' ').nth(1) != Some("200") {
            bail!("{status}");
        }
        Ok(resp[split + 4..].to_vec())
    })
    .await
    .with_context(|| format!("no answer from {addr} within {limit:?}"))?
}

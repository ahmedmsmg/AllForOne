//! Just enough HTTP/1.1 for status pages and the agent API: GET and POST, one request per
//! connection, small bodies.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::timeout,
};
use tracing::debug;

const MAX_REQUEST: usize = 16 << 10;
const MAX_BODY: usize = 1 << 20;
const IO_TIMEOUT: Duration = Duration::from_secs(5);

pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    pub body: Vec<u8>,
    pub peer: SocketAddr,
}

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
        Self::error(404, "not found")
    }

    pub fn error(status: u16, message: &str) -> Self {
        Self { status, content_type: "text/plain", body: format!("{message}\n").into_bytes() }
    }
}

/// Answers requests on `listener` with `handler` until the task is dropped.
pub async fn serve(listener: TcpListener, handler: impl Fn(&Request) -> Response + Send + Sync + 'static) {
    let handler = Arc::new(handler);
    loop {
        let Ok((sock, peer)) = listener.accept().await else { continue };
        let handler = handler.clone();
        tokio::spawn(async move {
            if let Err(e) = respond(sock, peer, &*handler).await {
                debug!("http request from {peer}: {e:#}");
            }
        });
    }
}

async fn respond(
    mut sock: TcpStream,
    peer: SocketAddr,
    handler: &(impl Fn(&Request) -> Response + ?Sized),
) -> Result<()> {
    let mut buf = Vec::with_capacity(1024);
    let req = timeout(IO_TIMEOUT, async {
        let end = loop {
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
            ensure!(buf.len() < MAX_REQUEST, "request headers too large");
            ensure!(sock.read_buf(&mut buf).await? > 0, "connection closed mid-request");
        };
        let head = String::from_utf8_lossy(&buf[..end]).into_owned();
        let mut lines = head.lines();
        let mut parts = lines.next().unwrap_or_default().split(' ');
        let (method, target) = (parts.next().unwrap_or_default(), parts.next().unwrap_or_default());
        let len = lines
            .filter_map(|l| l.split_once(':'))
            .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.trim().parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        ensure!(len <= MAX_BODY, "request body too large");
        let mut body = buf.split_off(end);
        while body.len() < len {
            ensure!(sock.read_buf(&mut body).await? > 0, "connection closed mid-body");
        }
        body.truncate(len);
        Ok(Request {
            method: method.to_string(),
            path: target.split('?').next().unwrap_or(target).to_string(),
            body,
            peer,
        })
    })
    .await
    .context("timed out reading request")??;

    let resp = handler(&req);
    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
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
    request(addr, "GET", path, &[], limit).await
}

/// Sends `body` to `path` on `addr` and returns the body of a 200 response.
pub async fn post(addr: SocketAddr, path: &str, body: &[u8], limit: Duration) -> Result<Vec<u8>> {
    request(addr, "POST", path, body, limit).await
}

async fn request(addr: SocketAddr, method: &str, path: &str, body: &[u8], limit: Duration) -> Result<Vec<u8>> {
    timeout(limit, async {
        let mut sock = TcpStream::connect(addr).await?;
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        sock.write_all(head.as_bytes()).await?;
        sock.write_all(body).await?;
        let mut resp = Vec::new();
        sock.read_to_end(&mut resp).await?;
        let split = resp.windows(4).position(|w| w == b"\r\n\r\n").context("malformed HTTP response")?;
        let status = String::from_utf8_lossy(&resp[..split]).lines().next().unwrap_or_default().to_string();
        if status.split(' ').nth(1) != Some("200") {
            let detail = String::from_utf8_lossy(&resp[split + 4..]).trim().to_string();
            bail!("{status}: {detail}");
        }
        Ok(resp[split + 4..].to_vec())
    })
    .await
    .with_context(|| format!("no answer from {addr} within {limit:?}"))?
}

//! QUIC transport: one endpoint per process acting as both server and client, one persistent
//! connection and one long-lived unidirectional stream per destination, length-prefixed frames.
//!
//! TLS certificates are self-signed and not verified yet: this gives encryption in transit but
//! no peer authentication. Identity comes with the security phase.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result};
use quinn::{
    ClientConfig, Connection, Endpoint, RecvStream, ServerConfig, TransportConfig,
    congestion::CubicConfig,
    crypto::rustls::{QuicClientConfig, QuicServerConfig},
};
use rustls::{
    DigitallySignedStruct, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature},
    pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime},
};
use tokio::{
    runtime::Handle,
    sync::mpsc,
    time::{Instant, sleep_until},
};
use tracing::{debug, warn};

use crate::Msg;

const ALPN: &[u8] = b"afo/1";
const SERVER_NAME: &str = "afo";
const MAX_FRAME: usize = 256 << 20;

#[derive(Debug, Clone, Default)]
pub struct NetOpts {
    /// Artificial one-way delay added to every outgoing frame, to emulate a WAN hop locally.
    pub sim_delay: Duration,
}

/// A received message and the address of the peer that sent it.
pub type Inbound = (SocketAddr, Msg);

pub struct Net {
    endpoint: Endpoint,
    peers: Mutex<HashMap<SocketAddr, mpsc::UnboundedSender<(Instant, Vec<u8>)>>>,
    opts: NetOpts,
    rt: Handle,
}

impl Net {
    /// Binds `addr` and starts accepting peers. Must be called inside a Tokio runtime.
    pub fn bind(addr: SocketAddr, opts: NetOpts) -> Result<(Arc<Net>, mpsc::UnboundedReceiver<Inbound>)> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut endpoint = Endpoint::server(server_config(provider.clone())?, addr)
            .with_context(|| format!("binding {addr}"))?;
        endpoint.set_default_client_config(client_config(provider)?);

        let (tx, rx) = mpsc::unbounded_channel();
        let ep = endpoint.clone();
        tokio::spawn(async move {
            while let Some(incoming) = ep.accept().await {
                let tx = tx.clone();
                tokio::spawn(async move {
                    match incoming.await {
                        Ok(conn) => serve_connection(conn, tx).await,
                        Err(e) => warn!("incoming connection failed: {e}"),
                    }
                });
            }
        });

        let net = Arc::new(Net { endpoint, peers: Mutex::new(HashMap::new()), opts, rt: Handle::current() });
        Ok((net, rx))
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// Queues `msg` for `to` without blocking; usable from non-async threads. Connects lazily,
    /// and frames queued while the connection is being set up are sent once it is.
    pub fn send(&self, to: SocketAddr, msg: &Msg) -> Result<()> {
        let frame = msg.to_frame()?;
        let mut peers = self.peers.lock().unwrap();
        let tx = peers.entry(to).or_insert_with(|| {
            let (tx, rx) = mpsc::unbounded_channel();
            let endpoint = self.endpoint.clone();
            let delay = self.opts.sim_delay;
            self.rt.spawn(async move {
                if let Err(e) = write_frames(endpoint, to, rx, delay).await {
                    warn!("link to {to} closed: {e:#}");
                }
            });
            tx
        });
        if tx.send((Instant::now(), frame)).is_err() {
            // The writer died; drop it so the next send reconnects, and report this one lost.
            peers.remove(&to);
            anyhow::bail!("link to {to} is down");
        }
        Ok(())
    }
}

async fn write_frames(
    endpoint: Endpoint,
    to: SocketAddr,
    mut rx: mpsc::UnboundedReceiver<(Instant, Vec<u8>)>,
    delay: Duration,
) -> Result<()> {
    let conn = endpoint.connect(to, SERVER_NAME)?.await.with_context(|| format!("connecting to {to}"))?;
    debug!("connected to {to}");
    let mut stream = conn.open_uni().await?;
    while let Some((queued_at, frame)) = rx.recv().await {
        if !delay.is_zero() {
            // Every frame gets the same delay, so order is preserved and back-to-back frames
            // overlap their waits instead of adding up.
            sleep_until(queued_at + delay).await;
        }
        stream.write_all(&frame).await?;
    }
    Ok(())
}

async fn serve_connection(conn: Connection, tx: mpsc::UnboundedSender<Inbound>) {
    let peer = conn.remote_address();
    debug!("accepted {peer}");
    while let Ok(stream) = conn.accept_uni().await {
        let tx = tx.clone();
        tokio::spawn(async move {
            if let Err(e) = read_frames(peer, stream, tx).await {
                debug!("stream from {peer} ended: {e:#}");
            }
        });
    }
}

async fn read_frames(peer: SocketAddr, mut stream: RecvStream, tx: mpsc::UnboundedSender<Inbound>) -> Result<()> {
    let mut len = [0u8; 4];
    let mut body = Vec::new();
    loop {
        stream.read_exact(&mut len).await?;
        let n = u32::from_le_bytes(len) as usize;
        anyhow::ensure!(n <= MAX_FRAME, "frame of {n} bytes exceeds limit");
        body.resize(n, 0);
        stream.read_exact(&mut body).await?;
        let msg = Msg::from_body(&body)?;
        if tx.send((peer, msg)).is_err() {
            return Ok(());
        }
    }
}

fn transport_config() -> Arc<TransportConfig> {
    let mut t = TransportConfig::default();
    t.keep_alive_interval(Some(Duration::from_secs(5)));
    t.max_idle_timeout(Some(Duration::from_secs(60).try_into().unwrap()));
    // Large windows so a prefill burst (megabytes of activations) isn't throttled by flow control.
    t.stream_receive_window((32u32 << 20).into());
    t.receive_window((64u32 << 20).into());
    t.send_window(64 << 20);
    // A larger initial congestion window lets the first prefill go out without waiting on
    // slow-start round trips.
    let mut cc = CubicConfig::default();
    cc.initial_window(1 << 20);
    t.congestion_controller_factory(Arc::new(cc));
    Arc::new(t)
}

fn server_config(provider: Arc<CryptoProvider>) -> Result<ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(vec![SERVER_NAME.into()])?;
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));
    let mut tls = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.der().clone()], key)?;
    tls.alpn_protocols = vec![ALPN.to_vec()];
    let mut cfg = ServerConfig::with_crypto(Arc::new(QuicServerConfig::try_from(tls)?));
    cfg.transport_config(transport_config());
    Ok(cfg)
}

fn client_config(provider: Arc<CryptoProvider>) -> Result<ClientConfig> {
    let mut tls = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyCert(provider)))
        .with_no_client_auth();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    let mut cfg = ClientConfig::new(Arc::new(QuicClientConfig::try_from(tls)?));
    cfg.transport_config(transport_config());
    Ok(cfg)
}

/// Accepts any certificate but still checks handshake signatures, so the connection is encrypted
/// to whoever holds the presented key. Replaced by real node identities in the security phase.
#[derive(Debug)]
struct AcceptAnyCert(Arc<CryptoProvider>);

impl ServerCertVerifier for AcceptAnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

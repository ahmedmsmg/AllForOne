//! Pipeline node: runs one contiguous block of layers and pushes results to the next hop.
//! It also answers `GET /status` over TCP on the same port number, for `afo-monitor`.

mod status;

use std::{collections::HashMap, net::SocketAddr, ops::Range, path::PathBuf, sync::{Arc, Mutex}, time::{Duration, Instant}};

use afo_codec::CodecKind;
use afo_engine::{CandleEngine, Engine, LoadOpts, Shard};
use afo_proto::{Forward, HopStat, Inbound, Msg, Net, NetOpts, http, status::STATUS_PATH};
use anyhow::{Context, Result};
use clap::Parser;
use status::{Activity, Info};
use tokio::{net::TcpListener, sync::mpsc::UnboundedReceiver};
use tracing::{info, warn};

#[derive(Parser)]
#[command(about = "Run a block of model layers as one stage of the pipeline")]
struct Args {
    /// GGUF model file (only this node's layers are loaded from it).
    #[arg(long)]
    model: PathBuf,
    /// Layers to run, half-open: `0..8` runs blocks 0 through 7.
    #[arg(long, value_parser = afo_engine::parse_range)]
    layers: Range<usize>,
    #[arg(long, default_value = "0.0.0.0:7100")]
    listen: SocketAddr,
    /// cpu | cuda[:N] | metal
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 4096)]
    max_seq: usize,
    /// Extra one-way delay on every outgoing frame, to emulate a WAN link locally.
    #[arg(long, default_value_t = 0)]
    sim_delay_ms: u64,
}

struct Session {
    route: Vec<SocketAddr>,
    hop: u32,
}

impl Session {
    fn next(&self) -> SocketAddr {
        self.route[self.hop as usize + 1]
    }

    fn client(&self) -> SocketAddr {
        *self.route.last().unwrap()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    ).init();
    let args = Args::parse();

    let device = afo_engine::device(&args.device)?;
    let t = Instant::now();
    let opts = LoadOpts { layers: args.layers.clone(), embed: false, head: false, max_seq: args.max_seq };
    let shard = Shard::load(&args.model, &opts, &device)?;
    info!("loaded layers {:?} of {} in {:.1?}", args.layers, shard.cfg.n_layer, t.elapsed());

    let (net, rx) = Net::bind(args.listen, NetOpts { sim_delay: Duration::from_millis(args.sim_delay_ms) })?;
    info!("listening on {}", net.local_addr()?);

    let activity = Arc::new(Mutex::new(Activity::new()));
    let info = Info {
        host: sysinfo::System::host_name().unwrap_or_default(),
        model: args.model.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
        n_layer: shard.cfg.n_layer,
        layers: args.layers.clone(),
        device: args.device.clone(),
    };
    let resources = Arc::default();
    status::spawn_sampler(&args.device, Arc::clone(&resources));
    // Status is a convenience: a node without it still serves the ring.
    match TcpListener::bind(net.local_addr()?).await {
        Ok(listener) => {
            info!("status on http://{}{STATUS_PATH}", listener.local_addr()?);
            let activity = activity.clone();
            tokio::spawn(http::serve(listener, move |req| match req.path.as_str() {
                STATUS_PATH => {
                    let status = activity.lock().unwrap().snapshot(&info, &resources.lock().unwrap());
                    http::Response::json(serde_json::to_string(&status).expect("status serializes"))
                }
                _ => http::Response::not_found(),
            }));
        }
        Err(e) => warn!("no status endpoint: {e}"),
    }

    // Compute runs on its own thread so blocking matmuls never stall the network tasks.
    let worker = std::thread::spawn(move || serve(CandleEngine::new(shard), net, rx, activity));
    tokio::task::spawn_blocking(move || worker.join()).await?.expect("worker panicked")
}

fn serve(mut engine: impl Engine, net: Arc<Net>, mut rx: UnboundedReceiver<Inbound>, activity: Arc<Mutex<Activity>>) -> Result<()> {
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    while let Some((from, msg)) = rx.blocking_recv() {
        match msg {
            Msg::Open { session, route, hop } => {
                if route.len() < hop as usize + 2 {
                    warn!("session {session}: malformed route from {from}");
                    continue;
                }
                let s = Session { route: route.clone(), hop };
                net.send(s.next(), &Msg::Open { session, route, hop: hop + 1 })?;
                info!("session {session}: hop {hop}, next {}", s.next());
                sessions.insert(session, s);
                activity.lock().unwrap().sessions = sessions.len();
            }
            Msg::Forward(f) => {
                let Some(s) = sessions.get(&f.session) else {
                    warn!("forward for unknown session {} from {from}", f.session);
                    continue;
                };
                let (session, step) = (f.session, f.step);
                match run_step(&mut engine, f) {
                    Ok(out) => {
                        let busy = out.hops.last().map_or(0, |h| h.busy_us);
                        activity.lock().unwrap().record(out.n_tokens, Duration::from_micros(busy as u64));
                        net.send(s.next(), &Msg::Forward(out))?
                    }
                    Err(e) => {
                        activity.lock().unwrap().error();
                        warn!("session {session} step {step}: {e:#}");
                        let message = format!("{e:#}");
                        net.send(s.client(), &Msg::Error { session, hop: s.hop, message })?;
                    }
                }
            }
            Msg::Close { session } => {
                engine.close(session);
                if let Some(s) = sessions.remove(&session) {
                    net.send(s.next(), &Msg::Close { session })?;
                    activity.lock().unwrap().sessions = sessions.len();
                    info!("session {session}: closed");
                }
            }
            Msg::Error { session, .. } => warn!("unexpected error message for session {session} from {from}"),
        }
    }
    Ok(())
}

fn run_step(engine: &mut impl Engine, mut f: Forward) -> Result<Forward> {
    let t = Instant::now();
    let codec = CodecKind::from_u8(f.codec)?;
    let (n, d) = (f.n_tokens as usize, engine.hidden_size());
    let hidden = afo_codec::decode(codec, &f.payload, n, d)?;
    let out = engine.forward(f.session, f.pos as usize, &hidden, n).context("forward pass")?;
    f.payload = afo_codec::encode(codec, &out, d);
    f.hops.push(HopStat { busy_us: t.elapsed().as_micros() as u32 });
    Ok(f)
}

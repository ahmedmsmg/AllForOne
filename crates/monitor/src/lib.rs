//! Live dashboard for a running ring: polls each node's `/status` and serves a web page showing
//! which nodes are up, what they run, their resource use, and the generation rate.

use std::{
    collections::VecDeque,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use afo_proto::{
    http::{self, Response},
    status::{NodeStatus, STATUS_PATH},
};
use serde::Serialize;
use tokio::{net::TcpListener, time::MissedTickBehavior};

const PAGE: &str = include_str!("index.html");
/// Generation-rate samples kept for the chart: five minutes at the default interval.
const HISTORY: usize = 300;

#[derive(Clone, Serialize)]
struct NodeView {
    addr: SocketAddr,
    online: bool,
    /// Time to fetch the status, which includes a TCP handshake: about two network round trips.
    fetch_ms: Option<f64>,
    error: Option<String>,
    /// Latest status; kept after the node goes offline so the page can still name it.
    status: Option<NodeStatus>,
    #[serde(skip)]
    last_ok: Option<Instant>,
    /// Seconds since the status was last fetched successfully.
    seen_s_ago: Option<f64>,
}

#[derive(Serialize)]
struct Snapshot<'a> {
    interval_ms: u64,
    nodes: &'a [NodeView],
    /// (unix seconds, ring tokens/s)
    history: &'a VecDeque<(f64, f64)>,
}

struct State {
    nodes: Vec<NodeView>,
    history: VecDeque<(f64, f64)>,
}

/// Polls `nodes` every `interval` and serves the dashboard on `listener` until dropped.
pub async fn serve(listener: TcpListener, nodes: Vec<SocketAddr>, interval: Duration) {
    let blank = |addr| NodeView { addr, online: false, fetch_ms: None, error: None, status: None, last_ok: None, seen_s_ago: None };
    let state = Arc::new(Mutex::new(State { nodes: nodes.iter().map(|&a| blank(a)).collect(), history: VecDeque::new() }));
    tokio::spawn(poll(nodes, state.clone(), interval));
    http::serve(listener, move |req| match req.path.as_str() {
        "/" | "/index.html" => Response::html(PAGE),
        "/api/status" => {
            let s = state.lock().unwrap();
            let snap = Snapshot { interval_ms: interval.as_millis() as u64, nodes: &s.nodes, history: &s.history };
            Response::json(serde_json::to_string(&snap).expect("snapshot serializes"))
        }
        _ => Response::not_found(),
    })
    .await;
}

async fn poll(addrs: Vec<SocketAddr>, state: Arc<Mutex<State>>, interval: Duration) {
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let limit = interval.max(Duration::from_secs(2));
    loop {
        tick.tick().await;
        let fetches: Vec<_> = addrs
            .iter()
            .map(|&addr| {
                tokio::spawn(async move {
                    let t = Instant::now();
                    let status = async {
                        let body = http::get(addr, STATUS_PATH, limit).await?;
                        Ok::<NodeStatus, anyhow::Error>(serde_json::from_slice(&body)?)
                    }
                    .await;
                    (status, t.elapsed())
                })
            })
            .collect();
        let mut results = Vec::with_capacity(fetches.len());
        for f in fetches {
            results.push(f.await.expect("status fetch task panicked"));
        }

        let now = Instant::now();
        let mut s = state.lock().unwrap();
        for (view, (status, took)) in s.nodes.iter_mut().zip(results) {
            match status {
                Ok(st) => {
                    view.online = true;
                    view.fetch_ms = Some(took.as_secs_f64() * 1e3);
                    view.error = None;
                    view.status = Some(st);
                    view.last_ok = Some(now);
                }
                Err(e) => {
                    view.online = false;
                    view.fetch_ms = None;
                    view.error = Some(format!("{e:#}"));
                }
            }
            view.seen_s_ago = view.last_ok.map(|t| now.duration_since(t).as_secs_f64());
        }
        // Every generated token passes every node, so any live node's rate is the ring's.
        let rate = s
            .nodes
            .iter()
            .filter(|n| n.online)
            .filter_map(|n| n.status.as_ref().map(|st| st.decode_tok_s))
            .fold(0., f64::max);
        let unix = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        s.history.push_back((unix, rate));
        if s.history.len() > HISTORY {
            s.history.pop_front();
        }
    }
}

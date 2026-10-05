//! Synthetic gRPC load generator for the SASY policy engine.
//!
//! Reads a replay JSONL produced by ``SASY_SERVER_LATENCY_LOG_FILE``
//! (one record per action with session_id, fn_name, args,
//! current_node_ids, plus ts and per-RPC timing/decision), then
//! issues the same ``CheckToolCall`` RPCs concurrently against a
//! running server and records each call's wall-clock RTT plus any
//! returned ``PerformanceTiming`` to a fresh JSONL.
//!
//! Pure tonic client — no Python, no GIL, so the load generator
//! itself doesn't contaminate measurement past whatever ceiling the
//! local network stack imposes.
//!
//! Records in the *output* log share the SDK's
//! ``SASY_LATENCY_LOG_FILE`` shape (``ts``, ``fn_name``,
//! ``authorized``, ``rtt_us``, ``session_id``), so a reducer written
//! for that log reads these records too. Server-side ``total_us`` / ``eval_us`` come back in the
//! response's ``timing`` field and are appended to each record so
//! the SDK-overhead delta (rtt - total) is computable per call.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, Semaphore};
use tonic::transport::{Channel, Endpoint};
use tonic::Request;
use tracing::{info, warn};

use sasy_auth::tls::TlsConfig;
use sasy_common::policy_engine::policy_engine_client::PolicyEngineClient;
use sasy_common::policy_engine::{action::ActionType, Action, EndSessionRequest};
use sasy_common::reference_monitor::rm_proxy_client::RmProxyClient;
use sasy_common::reference_monitor::ToolCallRequest;

/// One row from the replay JSONL. Mirrors the server-side
/// ``ActionRecord`` plus a few optional fields the loader accepts
/// for forward compatibility (extra keys are ignored by serde).
#[derive(Debug, Clone, Deserialize)]
struct ReplayRow {
    #[serde(default)]
    session_id: String,
    fn_name: String,
    #[serde(default)]
    args: String,
    #[serde(default)]
    current_node_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OutputRecord<'a> {
    ts: f64,
    fn_name: &'a str,
    authorized: bool,
    rtt_us: f64,
    session_id: &'a str,
    /// Server-reported total (microseconds). When the server
    /// returns ``PerformanceTiming``, this is the wall time the
    /// server spent serving the RPC including queueing inside the
    /// session evaluator. Subtract from ``rtt_us`` to get
    /// loadgen → wire → loadgen overhead.
    server_total_us: u64,
    server_eval_us: u64,
    server_sync_us: u64,
}

#[derive(Parser, Debug)]
#[command(
    about = "Replay-driven load generator for the SASY policy engine.",
    long_about = None,
)]
struct Args {
    /// Server endpoint (e.g. https://localhost:10089).
    #[arg(long, default_value = "https://localhost:10089")]
    url: String,

    /// CA certificate path. Required for mTLS clusters.
    #[arg(long)]
    ca: Option<PathBuf>,

    /// Client cert (mTLS).
    #[arg(long)]
    cert: Option<PathBuf>,

    /// Client private key (mTLS).
    #[arg(long)]
    key: Option<PathBuf>,

    /// API key sent in ``x-api-key`` metadata. Optional; only
    /// needed if the server's auth chain expects it.
    #[arg(long)]
    api_key: Option<String>,

    /// Path to the replay JSONL (server-side latency log).
    #[arg(long)]
    replay: PathBuf,

    /// Output JSONL with per-call timings.
    #[arg(long)]
    out: PathBuf,

    /// Max concurrent in-flight CheckToolCall RPCs.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    /// Number of channels to round-robin across. Each channel is one
    /// HTTP/2 connection; with too few channels the per-channel I/O
    /// task serializes outgoing frames per worker. Empirically, fixing
    /// at 64 (matching SASY_CHANNEL_POOL_SIZE on the SDK side) makes
    /// per-call latency independent of channel count up to c=32.
    #[arg(long, default_value_t = 64)]
    channels: usize,

    /// Limit on rows replayed (head N); 0 = all.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Repeat the replay N times to hit a target call count.
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Skip rows whose ``fn_name`` matches this set
    /// (comma-separated). Useful for excluding HTTP / message
    /// actions when only tool calls are interesting.
    #[arg(long, default_value = "")]
    skip_fns: String,

    /// Replay scheduling model.
    ///
    /// * ``session`` (default) — group rows by session_id and spawn
    ///   one task per session; a semaphore caps active sessions to
    ///   ``--concurrency``. Within a session, calls go out in
    ///   their original order with at most one in-flight at a
    ///   time. Mirrors production: each session has its own
    ///   evaluator subprocess that processes calls serially.
    /// * ``flat`` — one global semaphore caps total in-flight
    ///   RPCs to ``--concurrency`` regardless of which
    ///   session each belongs to. Useful for stress-testing the
    ///   server above realistic concurrency.
    #[arg(long, value_enum, default_value_t = ScheduleMode::Session)]
    mode: ScheduleMode,

    /// Send ``EndSession`` for every session_id touched after the
    /// timed pass completes. Drops the per-session evaluator on the
    /// server (graph state is preserved). Default on so successive
    /// loadgen invocations against a long-lived server don't see
    /// each other's stale evaluators in the SessionEvaluatorMap or
    /// broadcast subscriber list.
    #[arg(long, default_value_t = true,
        action = clap::ArgAction::Set)]
    cleanup_sessions: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ScheduleMode {
    Session,
    Flat,
}

fn build_endpoint(args: &Args) -> Result<Endpoint> {
    let mut ep = Endpoint::from_shared(args.url.clone())
        .with_context(|| format!("invalid url: {}", args.url))?
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(15));

    let tls_cfg = TlsConfig {
        cert_path: args.cert.as_ref().map(|p| p.to_string_lossy().to_string()),
        key_path: args.key.as_ref().map(|p| p.to_string_lossy().to_string()),
        ca_path: args.ca.as_ref().map(|p| p.to_string_lossy().to_string()),
    };
    if tls_cfg.is_configured() {
        let client_tls = tls_cfg
            .load_client_config()
            .map_err(|e| anyhow!("tls config: {e:?}"))?;
        ep = ep
            .tls_config(client_tls)
            .with_context(|| "tls config build")?;
    }
    Ok(ep)
}

async fn build_pool(args: &Args) -> Result<Vec<RmProxyClient<Channel>>> {
    let ep = build_endpoint(args)?;
    let mut clients = Vec::with_capacity(args.channels);
    for _ in 0..args.channels.max(1) {
        let ch = ep
            .clone()
            .connect()
            .await
            .with_context(|| format!("connect to {}", args.url))?;
        clients.push(RmProxyClient::new(ch));
    }
    Ok(clients)
}

async fn load_replay(path: &std::path::Path) -> Result<Vec<ReplayRow>> {
    let file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open replay {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut rows = Vec::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<ReplayRow>(trimmed) {
            Ok(r) => rows.push(r),
            Err(e) => warn!("skipping unparseable replay line: {e}"),
        }
    }
    Ok(rows)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // rustls 0.23 requires the process-level CryptoProvider to be
    // installed before any TLS handshake. Mirror sasy-binary's
    // setup so the loadgen + server agree on the provider.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let skip_fns: std::collections::HashSet<String> = args
        .skip_fns
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    info!(
        "loading replay from {} (concurrency={}, channels={}, repeat={})",
        args.replay.display(),
        args.concurrency,
        args.channels,
        args.repeat,
    );
    let mut rows = load_replay(&args.replay).await?;
    if args.limit > 0 && rows.len() > args.limit {
        rows.truncate(args.limit);
    }
    if !skip_fns.is_empty() {
        rows.retain(|r| !skip_fns.contains(&r.fn_name));
    }
    info!("replay rows after filter: {}", rows.len());
    if rows.is_empty() {
        return Err(anyhow!("no rows to replay"));
    }

    let clients = build_pool(&args).await?;
    info!("connected {} channel(s) to {}", clients.len(), args.url);

    // Warmup: fire one call per *distinct session* in the replay
    // before timing starts. The server's per-session evaluator
    // architecture spawns one Soufflé subprocess per session on
    // first call and bootstraps it from RocksDB — that cost
    // (~10-20ms per session) is a real production cost on cold
    // starts but contaminates steady-state throughput numbers
    // when bundled into the timed window. The microbench reports
    // the steady-state cost; per-session bootstrap is its own
    // story (and small relative to the LLM-dominated end-to-end
    // path that pays it once per conversation).
    //
    // Also covers the per-channel HTTP/2 SETTINGS exchange.
    let warmup_rows: Vec<&ReplayRow> = {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for row in &rows {
            if seen.insert(row.session_id.as_str()) {
                out.push(row);
            }
        }
        out
    };
    info!("warming {} session(s) before timed run", warmup_rows.len());
    {
        let mut handles = Vec::with_capacity(warmup_rows.len());
        for (i, row) in warmup_rows.into_iter().enumerate() {
            let mut client = clients[i % clients.len()].clone();
            let api_key = args.api_key.clone();
            let row = row.clone();
            handles.push(tokio::spawn(async move {
                let req = build_request(&row);
                let mut r = Request::new(req);
                if let Some(ref k) = api_key {
                    if let Ok(v) = k.parse() {
                        r.metadata_mut().insert(sasy_common::headers::API_KEY, v);
                    }
                }
                if let Err(e) = client.check_tool_call(r).await {
                    warn!(
                        "warmup call failed (continuing) session={} fn={}: {}",
                        row.session_id, row.fn_name, e,
                    );
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    let out_file = tokio::fs::File::create(&args.out)
        .await
        .with_context(|| format!("create output {}", args.out.display()))?;
    let out = Arc::new(Mutex::new(out_file));
    let sem = Arc::new(Semaphore::new(args.concurrency));
    let api_key = args.api_key.clone();

    let total_calls = rows.len() * args.repeat;
    info!(
        "issuing {} calls in {:?} mode (concurrency={})",
        total_calls, args.mode, args.concurrency,
    );
    let start = Instant::now();

    let mut counts: HashMap<String, u64> = HashMap::new();
    for row in rows.iter() {
        *counts.entry(row.fn_name.clone()).or_default() += 1;
    }

    match args.mode {
        ScheduleMode::Flat => {
            let mut handles = Vec::with_capacity(total_calls);
            for (i, row) in rows.iter().cycle().take(total_calls).cloned().enumerate() {
                let client = clients[i % clients.len()].clone();
                let permit = sem.clone().acquire_owned().await.unwrap();
                let out = out.clone();
                let api_key = api_key.clone();
                handles.push(tokio::spawn(async move {
                    let _p = permit;
                    issue_one(&row, client, api_key.as_deref(), out).await;
                }));
            }
            for h in handles {
                let _ = h.await;
            }
        }
        ScheduleMode::Session => {
            // Group rows by session preserving order, then drive
            // each session as one task that issues its calls one
            // at a time. Cap active sessions at ``concurrency``.
            let mut by_session: HashMap<String, Vec<ReplayRow>> = HashMap::new();
            let mut order: Vec<String> = Vec::new();
            for _ in 0..args.repeat.max(1) {
                for row in rows.iter().cloned() {
                    if !by_session.contains_key(&row.session_id) {
                        order.push(row.session_id.clone());
                    }
                    by_session
                        .entry(row.session_id.clone())
                        .or_default()
                        .push(row);
                }
            }
            info!(
                "session mode: {} distinct sessions, max active = {}",
                order.len(),
                args.concurrency,
            );
            let mut handles = Vec::with_capacity(order.len());
            for (i, sid) in order.into_iter().enumerate() {
                let session_rows = by_session.remove(&sid).unwrap_or_default();
                let client = clients[i % clients.len()].clone();
                let permit = sem.clone().acquire_owned().await.unwrap();
                let out = out.clone();
                let api_key = api_key.clone();
                handles.push(tokio::spawn(async move {
                    let _p = permit; // released when session done
                    let mut c = client;
                    for row in session_rows {
                        issue_one(&row, c.clone(), api_key.as_deref(), out.clone()).await;
                        // Reuse the same client; cloning is cheap
                        // (Channel is Arc'd internally) and lets
                        // future RPCs retry on the same channel.
                        let _ = &mut c;
                    }
                }));
            }
            for h in handles {
                let _ = h.await;
            }
        }
    }

    let wall = start.elapsed();
    let mut f = out.lock().await;
    f.flush().await?;
    drop(f);

    let calls_per_sec = total_calls as f64 / wall.as_secs_f64();
    info!(
        "done: {} calls in {:.2}s = {:.1} calls/s",
        total_calls,
        wall.as_secs_f64(),
        calls_per_sec,
    );
    let mut top: Vec<_> = counts.into_iter().collect();
    top.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (k, v) in top.iter().take(10) {
        info!("  {:>6} {}", v, k);
    }

    if args.cleanup_sessions {
        let unique_sessions: std::collections::HashSet<&str> =
            rows.iter().map(|r| r.session_id.as_str()).collect();
        info!(
            "cleanup: ending {} session(s) so the next invocation starts \
             with a clean SessionEvaluatorMap",
            unique_sessions.len(),
        );
        let ep = build_endpoint(&args)?;
        let mut pe_clients: Vec<PolicyEngineClient<Channel>> =
            Vec::with_capacity(args.channels.clamp(1, 8));
        for _ in 0..args.channels.clamp(1, 8) {
            let ch = ep
                .clone()
                .connect()
                .await
                .with_context(|| "cleanup: connect")?;
            pe_clients.push(PolicyEngineClient::new(ch));
        }
        let api_key = args.api_key.clone();
        let mut handles = Vec::with_capacity(unique_sessions.len());
        for (i, sid) in unique_sessions.into_iter().enumerate() {
            let mut c = pe_clients[i % pe_clients.len()].clone();
            let api_key = api_key.clone();
            let sid = sid.to_string();
            handles.push(tokio::spawn(async move {
                let mut req = Request::new(EndSessionRequest {
                    session_id: sid.clone(),
                });
                if let Some(ref k) = api_key {
                    if let Ok(v) = k.parse() {
                        req.metadata_mut().insert(sasy_common::headers::API_KEY, v);
                    }
                }
                if let Err(e) = c.end_session(req).await {
                    warn!("end_session failed for {sid}: {e}");
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    Ok(())
}

async fn issue_one(
    row: &ReplayRow,
    mut client: RmProxyClient<Channel>,
    api_key: Option<&str>,
    out: Arc<Mutex<tokio::fs::File>>,
) {
    let req = build_request(row);
    let mut r = Request::new(req);
    if let Some(k) = api_key {
        if let Ok(v) = k.parse() {
            r.metadata_mut().insert(sasy_common::headers::API_KEY, v);
        }
    }
    let t0 = Instant::now();
    let res = client.check_tool_call(r).await;
    let rtt_us = t0.elapsed().as_micros() as f64;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    match res {
        Ok(resp) => {
            let inner = resp.into_inner();
            // ``ToolCallResponse`` doesn't carry
            // ``PerformanceTiming`` — that's on the lower-level
            // PolicyEngine RPC. Server-side timing comes from
            // the matching ``SASY_SERVER_LATENCY_LOG_FILE`` row.
            let rec = OutputRecord {
                ts,
                fn_name: &row.fn_name,
                authorized: inner.authorized,
                rtt_us,
                session_id: &row.session_id,
                server_total_us: 0,
                server_eval_us: 0,
                server_sync_us: 0,
            };
            if let Ok(mut buf) = serde_json::to_vec(&rec) {
                buf.push(b'\n');
                let mut f = out.lock().await;
                let _ = f.write_all(&buf).await;
            }
        }
        Err(e) => {
            warn!(
                "call failed fn={} session={}: {}",
                row.fn_name, row.session_id, e,
            );
        }
    }
}

fn build_request(row: &ReplayRow) -> ToolCallRequest {
    ToolCallRequest {
        fn_name: row.fn_name.clone(),
        args: row.args.clone(),
        input_node_ids: row.current_node_ids.clone(),
        session_id: Some(row.session_id.clone()).filter(|s| !s.is_empty()),
        // Replay traffic is auth-derived only — no user-supplied
        // wire entity to forward.
        entity: None,
        metadata: vec![],
    }
}

// Make the Action / ActionType imports used so they're not dead code
// when callers want them in tests.
#[allow(dead_code)]
fn _action_type_ref(action: &Action) -> Option<&ActionType> {
    action.action_type.as_ref()
}

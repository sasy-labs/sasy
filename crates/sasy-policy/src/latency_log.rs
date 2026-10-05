//! Optional per-call latency log for the policy engine.
//!
//! When the env var ``SASY_SERVER_LATENCY_LOG_FILE`` is set, every
//! ``check_authorization`` call emits one JSONL record with the same
//! shape the Python SDK writes for client-side RTT, plus the
//! server-side breakdown (``sync_us``, ``flush_us``, ``eval_us``).
//! Used by the synthetic load generator and the airline microbench
//! to attribute time across SDK / wire / engine without parsing
//! debug logs.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Serialize;

/// One JSONL record per action in a CheckAuthorization RPC. Multiple
/// actions in the same RPC share ``request_id`` and the timing
/// breakdown (timings are per-RPC, not per-action). Replay consumers
/// can group on ``request_id`` to reconstruct the original RPC shape.
#[derive(Serialize)]
pub struct ActionRecord<'a> {
    pub ts: f64,
    pub request_id: u64,
    pub action_idx: u32,
    pub fn_name: &'a str,
    pub args: &'a str,
    pub current_node_ids: &'a [String],
    pub authorized: bool,
    pub total_us: u64,
    pub sync_us: u64,
    pub flush_us: u64,
    pub eval_us: u64,
    pub session_id: &'a str,
    pub backend: &'a str,
}

/// Lazily-opened file. ``None`` means logging is disabled (env var
/// unset or open() failed). The Mutex guards write ordering so
/// concurrent writers don't interleave partial records on platforms
/// where ``write(2)`` isn't atomic for the buffer size we use.
static LOG_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

fn log_file() -> Option<&'static Mutex<std::fs::File>> {
    LOG_FILE
        .get_or_init(|| {
            let path = std::env::var("SASY_SERVER_LATENCY_LOG_FILE").ok()?;
            if path.is_empty() {
                return None;
            }
            OpenOptions::new()
                .create(true)
                .append(true)
                .mode(0o644)
                .open(&path)
                .map(Mutex::new)
                .ok()
        })
        .as_ref()
}

/// Monotonic per-process request id. Wraps after 2^64 calls; not
/// guaranteed unique across server restarts but unique within one
/// run, which is all the replay loader needs.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Reserve a fresh request id. Each CheckAuthorization RPC should
/// call this once and reuse the value across all of its action
/// records.
pub fn next_request_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Whether the latency log is enabled. Cheap-ish (one OnceLock read);
/// callers can use this to skip the per-action serialization loop
/// when logging is off, since most installations leave the env var
/// unset and we don't want to pay for the iteration.
pub fn enabled() -> bool {
    log_file().is_some()
}

/// Append one JSONL record. No-op when the env var is unset.
pub fn log_action(rec: &ActionRecord<'_>) {
    let Some(file) = log_file() else { return };
    let mut buf = match serde_json::to_vec(rec) {
        Ok(b) => b,
        Err(_) => return,
    };
    buf.push(b'\n');
    let mut guard = file.lock();
    let _ = guard.write_all(&buf);
}

/// Wall-clock seconds since UNIX epoch as f64 — same shape the SDK
/// writes for client-side RTT records.
pub fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

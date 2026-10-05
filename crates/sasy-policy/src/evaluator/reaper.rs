//! The list of evaluator process groups this server is responsible for
//! killing, and the two ways it kills them.
//!
//! An evaluator subprocess is spawned into its own **process group** (a set of
//! processes the kernel lets you signal with one call), so that one signal
//! reaches the whole tree. That matters because under `bwrap` the process this
//! server holds a handle to is only the outer wrapper: the evaluator itself
//! runs as init inside a new PID namespace and does not die with the wrapper.
//! A functor spinning in a loop never reads its stdin either, so closing the
//! pipe does not stop it. Left alone it burns a core and holds its address
//! space for as long as the host lives.
//!
//! Sending that signal is the job of [`crate::evaluator::manager::EvaluatorProcess::kill`],
//! and the session evaluator calls it when a query overruns its stall window.
//! But a stall is not the only way an evaluator stops being wanted:
//!
//!   * the session can be **evicted while it is busy** — a forced policy
//!     rollout, `EndSession`, a session-scoped rebind, or the idle sweep — and
//!     eviction drops the evaluator instead of waiting for a stall that will
//!     now never be noticed;
//!   * the **server process itself can exit**, on Ctrl-C or a `SIGTERM` from a
//!     service manager, at which point no Rust destructor runs at all.
//!
//! This module covers both. Every live evaluator's process-group id is
//! registered here at spawn and removed when it is killed or reaped;
//! [`kill_all`] signals whatever is still listed. Drop handlers call it for
//! one group, and a signal handler installed on `SIGINT` / `SIGTERM` calls it
//! for all of them before letting the signal take its normal course.
//!
//! On a platform without POSIX signals everything here compiles to nothing.

#[cfg(unix)]
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// How many evaluator groups can be tracked at once.
///
/// One slot per live evaluator process, which is one per `(tenant, session)`
/// with a policy bound — well above what a single server holds open, and the
/// storage is a fixed array so that the signal handler needs no allocation.
#[cfg(unix)]
const MAX_TRACKED: usize = 1024;

/// The registered process-group ids; `0` marks a free slot.
///
/// A plain array of atomics rather than a `Mutex<HashSet<_>>` because a signal
/// handler must not take a lock: the signal can arrive on a thread that is
/// already holding it, and the handler would deadlock the process it is
/// supposed to be cleaning up after.
#[cfg(unix)]
static SLOTS: [AtomicU32; MAX_TRACKED] = [const { AtomicU32::new(0) }; MAX_TRACKED];

/// Groups that did not fit in [`SLOTS`]. Reported so an operator sees the
/// bound was reached rather than silently losing the cleanup.
#[cfg(unix)]
static UNTRACKED: AtomicUsize = AtomicUsize::new(0);

/// Start tracking `pgid` as a group to kill.
///
/// A no-op for `0`, which is what the spawn path reports when it could not
/// learn the child's pid.
#[cfg(unix)]
pub(crate) fn register(pgid: u32) {
    register_in(&SLOTS, pgid)
}

/// [`register`] against a given slot array. The tests own their own array so
/// that a shutdown test cannot kill another test's child.
#[cfg(unix)]
fn register_in(slots: &[AtomicU32], pgid: u32) {
    if pgid == 0 {
        return;
    }
    for slot in slots.iter() {
        if slot
            .compare_exchange(0, pgid, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
    UNTRACKED.fetch_add(1, Ordering::Relaxed);
    tracing::warn!(
        pgid,
        "more than {} evaluator processes are live; this one will not be killed on shutdown",
        MAX_TRACKED
    );
}

/// Stop tracking `pgid` — it has been killed and reaped, so its number may be
/// handed to an unrelated process at any moment and must never be signalled
/// again.
#[cfg(unix)]
pub(crate) fn forget(pgid: u32) {
    forget_in(&SLOTS, pgid)
}

/// [`forget`] against a given slot array.
#[cfg(unix)]
fn forget_in(slots: &[AtomicU32], pgid: u32) {
    if pgid == 0 {
        return;
    }
    for slot in slots.iter() {
        if slot
            .compare_exchange(pgid, 0, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

/// `SIGKILL` every registered group and clear the list.
///
/// Async-signal-safe: no allocation, no locks, and `kill(2)` is on the
/// permitted list. Each slot is taken with a swap, so a group is signalled
/// once even if a Drop handler and the signal handler run at the same moment.
#[cfg(unix)]
pub(crate) fn kill_all() {
    kill_all_in(&SLOTS)
}

/// [`kill_all`] against a given slot array.
#[cfg(unix)]
fn kill_all_in(slots: &[AtomicU32]) {
    for slot in slots.iter() {
        let pgid = slot.swap(0, Ordering::AcqRel);
        if pgid != 0 {
            // SAFETY: `kill(2)` with a negative pid signals the group whose
            // leader is `pgid`. The pid is registered only between spawn and
            // reap, so it still names this server's own child.
            unsafe {
                libc::kill(-(pgid as i32), libc::SIGKILL);
            }
        }
    }
}

/// Kill one registered group now, without waiting for it to be reaped.
///
/// The synchronous half of the async `Evaluator::kill`, for destructors, which
/// cannot await. The signal is what stops the evaluator burning a core; the
/// reap is left to the `Child` handle's own `kill_on_drop`.
#[cfg(unix)]
pub(crate) fn kill_group(pgid: u32) {
    kill_group_in(&SLOTS, pgid)
}

/// [`kill_group`] against a given slot array.
#[cfg(unix)]
fn kill_group_in(slots: &[AtomicU32], pgid: u32) {
    if pgid == 0 {
        return;
    }
    forget_in(slots, pgid);
    // SAFETY: as in `kill_all`.
    unsafe {
        libc::kill(-(pgid as i32), libc::SIGKILL);
    }
}

/// The handler installed on `SIGINT` and `SIGTERM`.
///
/// Kills the registered groups, then puts the signal's default action back and
/// re-raises it, so the server dies exactly as it would have without this
/// handler — same exit status, same behaviour for anything watching it. It
/// does not turn Ctrl-C into a graceful shutdown; it only makes sure the
/// sandboxed evaluators go with it.
#[cfg(unix)]
extern "C" fn on_fatal_signal(sig: libc::c_int) {
    kill_all();
    // SAFETY: `signal` and `raise` are async-signal-safe, and restoring the
    // default disposition before re-raising is the standard way to let a
    // fatal signal do what it was going to do.
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

/// Install the shutdown handler, once per process.
///
/// Called from the evaluator spawn path, so a server that never starts an
/// evaluator never changes its signal disposition.
///
/// A handler already installed by the embedding program is left alone: this
/// only replaces the kernel's default action. If something else owns Ctrl-C in
/// this process, that owner is responsible for the cleanup and we do not want
/// two handlers fighting over the same signal.
#[cfg(unix)]
pub(crate) fn install_shutdown_handler() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for sig in [libc::SIGINT, libc::SIGTERM] {
            // SAFETY: reading the current disposition and setting ours; the
            // handler itself is async-signal-safe.
            unsafe {
                let previous =
                    libc::signal(sig, on_fatal_signal as *const () as libc::sighandler_t);
                if previous != libc::SIG_DFL {
                    // Someone else's handler (or SIG_IGN): put it back.
                    libc::signal(sig, previous);
                }
            }
        }
    });
}

#[cfg(not(unix))]
pub(crate) fn register(_pgid: u32) {}
#[cfg(not(unix))]
pub(crate) fn forget(_pgid: u32) {}
#[cfg(not(unix))]
pub(crate) fn kill_group(_pgid: u32) {}
#[cfg(not(unix))]
pub(crate) fn install_shutdown_handler() {}

/// Is `pgid` on the list [`kill_all`] would signal? Tests only — what it
/// checks is that a pid stops being listed the moment it is reaped, since from
/// then on the number can name an unrelated process.
#[cfg(all(test, unix))]
pub(crate) fn is_registered(pgid: u32) -> bool {
    pgid != 0 && SLOTS.iter().any(|s| s.load(Ordering::Acquire) == pgid)
}

/// True once the process is gone or is a zombie waiting to be reaped.
///
/// Waiting for the reap is not the point: the leak these tests guard against
/// is a process still *running*, burning a core inside its sandbox. A zombie
/// holds no CPU and no memory. Shared with the evaluator and session tests.
#[cfg(all(test, unix))]
pub(crate) fn stopped_running(pid: i32) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        // SAFETY: signal 0 only tests for the process.
        if unsafe { libc::kill(pid, 0) } != 0 {
            return true; // gone entirely
        }
        let state = std::process::Command::new("ps")
            .args(["-o", "state=", "-p", &pid.to_string()])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        if state.is_empty() || state.starts_with('Z') {
            return true;
        }
        if std::time::Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// True once no process is left in group `pgid`, waiting up to three seconds.
///
/// The leak these tests guard against is a process still *running*, burning a
/// core inside its sandbox, so a zombie awaiting its reap counts as stopped:
/// `pgrep` does not list zombies as members of the group once they are gone
/// from the run queue, and the reap itself belongs to the `Child` handle.
/// Shared with the evaluator and session tests.
#[cfg(all(test, unix))]
pub(crate) fn group_is_empty(pgid: u32) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        let listed = std::process::Command::new("pgrep")
            .args(["-g", &pgid.to_string()])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        let live = listed
            .lines()
            .filter_map(|l| l.trim().parse::<i32>().ok())
            .any(|pid| !is_zombie(pid));
        if !live {
            return true;
        }
        if std::time::Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// True if `pid` has exited and is only waiting to be reaped.
#[cfg(all(test, unix))]
fn is_zombie(pid: i32) -> bool {
    std::process::Command::new("ps")
        .args(["-o", "state=", "-p", &pid.to_string()])
        .output()
        .map(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            s.is_empty() || s.starts_with('Z')
        })
        .unwrap_or(false)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    /// The tests keep their own slot array: `kill_all` empties whatever array
    /// it is given, and the real one holds the children of every other test in
    /// this binary.
    fn slots() -> Vec<AtomicU32> {
        (0..8).map(|_| AtomicU32::new(0)).collect()
    }

    /// A real process in its own group that will not exit on its own.
    fn spawn_sleeper() -> std::process::Child {
        use std::os::unix::process::CommandExt;
        let mut cmd = Command::new("sleep");
        cmd.arg("300").stdout(Stdio::null()).stderr(Stdio::null());
        cmd.process_group(0);
        cmd.spawn().expect("sleep must be spawnable")
    }

    /// True once the child is gone, waiting up to `within`.
    fn died(child: &mut std::process::Child, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    /// The shutdown path kills a registered group. This is what a Ctrl-C on a
    /// server with a busy evaluator has to do; the signal handler calls this
    /// same function.
    #[test]
    fn kill_all_kills_a_registered_group() {
        let slots = slots();
        let mut child = spawn_sleeper();
        register_in(&slots, child.id());
        kill_all_in(&slots);
        assert!(
            died(&mut child, Duration::from_secs(2)),
            "a registered evaluator group must not survive shutdown"
        );
    }

    /// A group that was already cleaned up is not signalled again: its pid can
    /// be reused by an unrelated process.
    #[test]
    fn a_forgotten_group_is_not_killed() {
        let slots = slots();
        let mut child = spawn_sleeper();
        register_in(&slots, child.id());
        forget_in(&slots, child.id());
        kill_all_in(&slots);
        assert!(
            !died(&mut child, Duration::from_millis(300)),
            "a group removed from the list must be left alone"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Env var that turns the helper test below into the child half of the
    /// signal test. Set only by [`a_sigterm_kills_the_evaluator_groups`].
    const CHILD_MARKER: &str = "SASY_REAPER_SIGNAL_CHILD";

    /// The child half of [`a_sigterm_kills_the_evaluator_groups`], and a no-op
    /// in an ordinary run.
    ///
    /// A signal disposition is process-wide and
    /// [`install_shutdown_handler`] installs once per process, so the handler
    /// cannot be tested inside the test binary without changing what every
    /// other test in it sees on Ctrl-C. This body therefore runs in a fresh
    /// process — the test binary re-executed with `CHILD_MARKER` set and this
    /// test named — where it registers a real group in the real slot array,
    /// installs the handler, tells its parent the group id, and waits to be
    /// signalled.
    // Nothing waits on the sleeper: this process is killed by the signal
    // under test, and the sleeper with it. Its parent test kills the group
    // again afterwards in case it is not.
    #[allow(clippy::zombie_processes)]
    #[test]
    fn shutdown_signal_child() {
        if std::env::var(CHILD_MARKER).is_err() {
            return;
        }
        let child = spawn_sleeper();
        register(child.id());
        install_shutdown_handler();
        println!("PGID {}", child.id());
        use std::io::Write;
        std::io::stdout().flush().expect("tell the parent the pgid");
        // The parent's SIGTERM ends this process; the sleep only bounds a
        // parent that never sends one.
        std::thread::sleep(Duration::from_secs(30));
        panic!("no signal arrived");
    }

    /// A `SIGTERM` to a server holding a live evaluator kills the evaluator's
    /// process group, and then still ends the server the way the signal
    /// would have on its own — killed by that same signal, not exited.
    ///
    /// This is the shutdown half of the guarantee: destructors do not run when
    /// a process is signalled, so without the handler a sandboxed evaluator
    /// would be left running after its server is gone.
    #[test]
    fn a_sigterm_kills_the_evaluator_groups() {
        use std::io::{BufRead, BufReader};
        use std::os::unix::process::ExitStatusExt;

        let mut server = Command::new(std::env::current_exe().expect("test binary path"))
            .args([
                "--exact",
                "evaluator::reaper::tests::shutdown_signal_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_MARKER, "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("re-exec the test binary as the server half");

        // Read the group id, which the child prints only after the handler is
        // installed — so the signal below cannot race the installation.
        let mut out = BufReader::new(server.stdout.take().expect("piped stdout"));
        let mut pgid = 0u32;
        let mut line = String::new();
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            line.clear();
            if out.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            // The marker shares its line with libtest's own progress output,
            // which prints the test's name with no newline before the body
            // runs — so look for the marker anywhere in the line.
            if let Some(at) = line.find("PGID ") {
                let digits: String = line[at + 5..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                pgid = digits.parse().expect("the child prints a pgid");
                break;
            }
        }
        assert!(pgid != 0, "the server half never reported its evaluator");

        // SAFETY: signalling a process this test spawned.
        unsafe {
            libc::kill(server.id() as i32, libc::SIGTERM);
        }

        let status = {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match server.try_wait() {
                    Ok(Some(s)) => break s,
                    _ if Instant::now() > deadline => {
                        let _ = server.kill();
                        // SAFETY: as above; do not leave the sleeper behind.
                        unsafe { libc::kill(-(pgid as i32), libc::SIGKILL) };
                        panic!("the signalled server never exited");
                    }
                    _ => std::thread::sleep(Duration::from_millis(20)),
                }
            }
        };

        let empty = group_is_empty(pgid);
        // SAFETY: as above. A no-op when the group is already gone; here so a
        // failing assertion below cannot leave a `sleep 300` running.
        unsafe { libc::kill(-(pgid as i32), libc::SIGKILL) };

        assert!(
            empty,
            "the evaluator group must not survive its server's SIGTERM"
        );
        assert_eq!(
            status.signal(),
            Some(libc::SIGTERM),
            "the server must still die of the signal it was sent, not exit \
             normally: {status:?}"
        );
    }

    /// The single-group kill used by destructors, which cannot await the async
    /// `Evaluator::kill`.
    #[test]
    fn kill_group_kills_just_that_group() {
        let slots = slots();
        let mut doomed = spawn_sleeper();
        let mut spared = spawn_sleeper();
        register_in(&slots, doomed.id());
        register_in(&slots, spared.id());
        kill_group_in(&slots, doomed.id());
        assert!(
            died(&mut doomed, Duration::from_secs(2)),
            "the named group must be killed"
        );
        assert!(
            !died(&mut spared, Duration::from_millis(300)),
            "no other group may be touched"
        );
        let _ = spared.kill();
        let _ = spared.wait();
        let _ = doomed.wait();
    }
}

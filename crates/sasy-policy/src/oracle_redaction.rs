//! The oracle-redaction switch.
//!
//! `@llm_check_fn(prompt, context)` hands two strings the policy assembled —
//! the context is usually a message's contents — to an external model
//! provider. Those two strings are the one place the engine sends recorded
//! content off the host, so they are scrubbed on the way out (see
//! [`crate::evaluator::ipc`]). This module holds the switch that turns the
//! scrubbing off.
//!
//! The setting is process-wide and read on the query path, so it lives in a
//! [`OnceLock`] written once at startup by [`set_oracle_redaction`]. A read
//! before that write answers `true`: a build that forgets to call the setter
//! redacts, rather than sending raw text because of a missing call.

use std::sync::OnceLock;

use tracing::warn;

/// The startup setting. `None` until [`set_oracle_redaction`] runs.
static ORACLE_REDACTION: OnceLock<bool> = OnceLock::new();

/// Publish the startup setting. The first call wins; a later one is ignored
/// and logged, because the switch is a property of the process, not of a
/// request.
pub fn set_oracle_redaction(enabled: bool) {
    if let Err(already) = ORACLE_REDACTION.set(enabled) {
        if already != enabled {
            warn!(
                installed = already,
                ignored = enabled,
                "the oracle-redaction switch was already set; keeping the first setting"
            );
        }
    }
}

/// Whether the oracle strings are scrubbed before they leave the process.
///
/// `true` when nothing was published, so an unwired caller is the safe one.
pub fn oracle_redaction_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = test_switch::forced() {
        return forced;
    }
    *ORACLE_REDACTION.get().unwrap_or(&true)
}

/// A test binary is one process, so one `OnceLock` cannot carry both settings.
/// This override sits in front of the lock in `cfg(test)` builds only — the
/// shipped binary has neither the flag nor the branch that reads it.
#[cfg(test)]
pub(crate) mod test_switch {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, MutexGuard};

    const UNSET: u8 = 0;
    const ON: u8 = 1;
    const OFF: u8 = 2;

    static FORCED: AtomicU8 = AtomicU8::new(UNSET);
    /// Held for the lifetime of a [`Forced`] so two tests never fight over the
    /// flag. `Mutex<()>` and not the value itself: the query path reads the
    /// flag from whatever thread the runtime polls it on.
    static SERIAL: Mutex<()> = Mutex::new(());

    pub(crate) fn forced() -> Option<bool> {
        match FORCED.load(Ordering::Relaxed) {
            ON => Some(true),
            OFF => Some(false),
            _ => None,
        }
    }

    /// Force the switch for as long as the returned value lives.
    pub(crate) struct Forced(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl Drop for Forced {
        fn drop(&mut self) {
            FORCED.store(UNSET, Ordering::Relaxed);
        }
    }

    /// Take the serialization lock without forcing anything.
    ///
    /// For the test of the unforced default: it must not run beside a test
    /// that forces the flag, or it would read that test's setting.
    pub(crate) fn unforced() -> Forced {
        let guard = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        FORCED.store(UNSET, Ordering::Relaxed);
        Forced(guard)
    }

    pub(crate) fn force(enabled: bool) -> Forced {
        // A test that panicked while holding it poisoned nothing that matters:
        // the flag is reset in `Drop`, which runs on unwind.
        let guard = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        FORCED.store(if enabled { ON } else { OFF }, Ordering::Relaxed);
        Forced(guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A build that never publishes a setting still redacts.
    ///
    /// Read through `oracle_redaction_enabled`, which is what the query path
    /// calls: asking the `OnceLock` directly, with the fall-back spelled out
    /// again in the test, would pass no matter what that function does.
    #[test]
    fn an_unset_switch_reads_as_on() {
        // No `force`, but the same lock the forcing tests take: this is the
        // bare default, which is what a build that never calls the setter
        // gets, and a forced flag would answer in its place.
        let _serial = test_switch::unforced();
        assert!(
            ORACLE_REDACTION.get().is_none(),
            "something in this test binary published a setting, so the unset default is \
             not what was measured here"
        );
        assert!(
            oracle_redaction_enabled(),
            "a switch nobody set must read as on"
        );
    }

    #[test]
    fn the_forced_switch_is_what_the_query_path_reads() {
        {
            let _off = test_switch::force(false);
            assert!(!oracle_redaction_enabled());
        }
        let _on = test_switch::force(true);
        assert!(oracle_redaction_enabled());
    }
}

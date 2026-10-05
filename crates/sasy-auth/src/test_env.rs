//! Serialised environment mutation for tests.
//!
//! The API key suffix rule reads process environment, and cargo runs a crate's
//! tests as threads of one process: two tests setting `SASY_API_KEY_SUFFIX`
//! concurrently would see each other's value. Every test that depends on those
//! variables takes [`EnvGuard::set`], which holds a crate-wide lock for the
//! test's lifetime and restores the previous values on drop.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Holds the environment lock and the values to restore.
pub struct EnvGuard {
    _guard: MutexGuard<'static, ()>,
    previous: HashMap<String, Option<String>>,
}

impl EnvGuard {
    /// Take the lock and apply `vars`; `None` clears the variable.
    ///
    /// A panicking test poisons the lock, which says nothing about the next
    /// test's environment (the guard's `Drop` still ran), so the poison is
    /// stepped over rather than propagated.
    pub fn set(vars: &[(&str, Option<&str>)]) -> Self {
        let guard = lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut previous = HashMap::new();
        for (name, value) in vars {
            previous.insert((*name).to_string(), std::env::var(name).ok());
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.previous {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

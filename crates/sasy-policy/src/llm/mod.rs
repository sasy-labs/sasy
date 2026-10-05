//! LLM Module
//!
//! Provides LLM-based semantic checks for Soufflé policy evaluation.
//! The main entry point is the `check()` function, which is called by the
//! `llm_check` functor in Soufflé policies (via the evaluator's oracle
//! callback).

pub mod cache;
pub mod client;
pub mod config;
pub mod service;

use std::sync::OnceLock;
use tracing::{info, warn};

use config::LlmConfig;
use service::LlmService;

/// Global LLM service instance (initialized once, then lock-free access)
static LLM_SERVICE: OnceLock<LlmService> = OnceLock::new();

/// Initialize the LLM service with configuration from environment variables.
/// This should be called once during application startup.
/// Subsequent calls are no-ops.
pub fn init() {
    LLM_SERVICE.get_or_init(|| {
        let config = LlmConfig::from_env();
        if config.enabled {
            let has_key = config.openai_api_key.is_some() || config.azure_api_key.is_some();
            info!(
                provider = ?config.provider,
                model = %config.model,
                has_api_key = has_key,
                valkey_url = %config.valkey_url,
                "LLM service enabled — @llm_check_fn oracle callbacks active"
            );
            if !has_key {
                warn!(
                    "LLM enabled but no API key set (OPENAI_API_KEY or AZURE_OPENAI_API_KEY). \
                     LLM calls will fail and @llm_check_fn will return false (fail-safe deny)."
                );
            }
        } else {
            info!("LLM service disabled (LLM_ENABLED != true). @llm_check_fn returns 0 (deny).");
        }
        LlmService::new(config)
    });
}

/// Initialize the LLM service with a custom configuration.
/// Useful for testing or custom setups.
/// Returns false if already initialized (first init wins).
pub fn init_with_config(config: LlmConfig) -> bool {
    LLM_SERVICE.set(LlmService::new(config)).is_ok()
}

/// Check a condition using the LLM.
/// This is the main entry point called by the Soufflé `llm_check` functor.
///
/// Concurrent calls are allowed - the underlying service uses channels
/// to dispatch requests to a dedicated async runtime.
///
/// Returns `false` (fail-safe deny) if:
/// - LLM service is not initialized
/// - LLM service is disabled
/// - LLM call times out
/// - LLM API returns an error
/// - LLM response cannot be parsed
pub fn check(prompt: &str, context: &str) -> bool {
    match LLM_SERVICE.get() {
        Some(service) => service.check_sync(prompt, context),
        None => {
            tracing::warn!("LLM service not initialized, returning false");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_returns_false_when_uninitialized() {
        // OnceLock may already be set from other tests in this process,
        // but a disabled service also returns false, so this always holds.
        let result = check("test prompt", "test context");
        assert!(!result);
    }

    #[test]
    fn init_with_disabled_config() {
        let config = config::LlmConfig::default(); // enabled = false
                                                   // init_with_config returns false if already initialized (OnceLock)
        let _ = init_with_config(config);
        // Either way, check should return false (disabled)
        assert!(!check("any", "any"));
    }
}

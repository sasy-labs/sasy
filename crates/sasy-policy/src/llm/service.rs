//! LLM Service
//!
//! Dedicated thread with its own tokio runtime to bridge the
//! evaluator's synchronous extern functions with async LLM calls.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;
use tracing::{debug, error, info, warn};

use super::cache::LlmCache;
use super::client::LlmClient;
use super::config::LlmConfig;

/// Request message sent to the LLM service thread
struct LlmRequest {
    prompt: String,
    context: String,
    response_tx: mpsc::Sender<bool>,
}

/// LLM service that runs in a dedicated thread
pub struct LlmService {
    request_tx: Sender<LlmRequest>,
    enabled: bool,
    timeout: Duration,
}

impl LlmService {
    /// Create a new LLM service from configuration
    pub fn new(config: LlmConfig) -> Self {
        if !config.enabled {
            info!("LLM service disabled");
            return Self {
                request_tx: mpsc::channel().0, // Dummy sender
                enabled: false,
                timeout: config.timeout,
            };
        }

        if !config.is_valid() {
            warn!("LLM configuration invalid, service disabled");
            return Self {
                request_tx: mpsc::channel().0,
                enabled: false,
                timeout: config.timeout,
            };
        }

        let (request_tx, request_rx) = mpsc::channel();
        let timeout = config.timeout;

        // Spawn dedicated thread with its own tokio runtime
        thread::spawn(move || {
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    error!("Failed to create tokio runtime for LLM service: {}", e);
                    return;
                }
            };

            rt.block_on(async {
                run_service_loop(request_rx, config).await;
            });
        });

        info!("LLM service started");

        Self {
            request_tx,
            enabled: true,
            timeout,
        }
    }

    /// Synchronous interface for the evaluator's extern function
    /// Returns false on any error (fail-safe deny)
    pub fn check_sync(&self, prompt: &str, context: &str) -> bool {
        if !self.enabled {
            debug!("LLM service disabled, returning false");
            return false;
        }

        let (response_tx, response_rx) = mpsc::channel();

        let request = LlmRequest {
            prompt: prompt.to_string(),
            context: context.to_string(),
            response_tx,
        };

        if self.request_tx.send(request).is_err() {
            error!("Failed to send request to LLM service");
            return false;
        }

        // Wait for response with timeout
        match response_rx.recv_timeout(self.timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                warn!("LLM request timed out after {:?}", self.timeout);
                false
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                error!("LLM service disconnected");
                false
            }
        }
    }
}

/// Main service loop running in the dedicated thread
async fn run_service_loop(request_rx: Receiver<LlmRequest>, config: LlmConfig) {
    // Initialize client
    let client = match LlmClient::new(&config) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to create LLM client: {}", e);
            // Drain requests and respond with false
            while let Ok(req) = request_rx.recv() {
                let _ = req.response_tx.send(false);
            }
            return;
        }
    };

    // Initialize cache (optional - failures are non-fatal)
    let cache = LlmCache::new(&config.valkey_url, config.cache_ttl_secs).ok();
    if cache.is_some() {
        info!("LLM cache connected to {}", config.valkey_url);
    } else {
        warn!("LLM cache not available, proceeding without caching");
    }

    info!(
        "LLM service loop started with provider: {:?}",
        config.provider
    );

    // Process requests
    while let Ok(request) = request_rx.recv() {
        let result =
            process_request(&client, cache.as_ref(), &request.prompt, &request.context).await;

        if request.response_tx.send(result).is_err() {
            debug!("Response receiver dropped");
        }
    }

    info!("LLM service loop ended");
}

/// Process a single LLM request with caching
async fn process_request(
    client: &LlmClient,
    cache: Option<&LlmCache>,
    prompt: &str,
    context: &str,
) -> bool {
    // Check cache first
    if let Some(cache) = cache {
        if let Some(cached) = cache.get(prompt, context).await {
            info!("[LLM] Cache HIT | result={}", cached);
            return cached;
        }
        info!("[LLM] Cache MISS");
    }

    // Call LLM
    let result = match client.check(prompt, context).await {
        Ok(r) => r,
        Err(e) => {
            error!("LLM API error: {}", e);
            return false; // Fail-safe
        }
    };

    // Cache result
    if let Some(cache) = cache {
        cache.set(prompt, context, result).await;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::config::LlmProvider;

    #[test]
    fn disabled_service_returns_false() {
        let config = LlmConfig::default();
        let service = LlmService::new(config);

        assert!(!service.enabled);
        assert!(!service.check_sync("test", "context"));
    }

    #[test]
    fn invalid_azure_config_disables_service() {
        let config = LlmConfig {
            enabled: true,
            provider: LlmProvider::Azure,
            // Missing required azure_api_key and azure_endpoint
            azure_api_key: None,
            azure_endpoint: None,
            ..LlmConfig::default()
        };
        assert!(!config.is_valid());

        let service = LlmService::new(config);
        assert!(!service.enabled);
        assert!(!service.check_sync("prompt", "context"));
    }

    #[test]
    fn enabled_openai_without_key_still_starts() {
        // OpenAI allows empty API key (for compatible services)
        let config = LlmConfig {
            enabled: true,
            provider: LlmProvider::OpenAI,
            openai_api_key: None,
            // Unreachable URL so it won't actually connect
            openai_base_url: Some("http://127.0.0.1:1".to_string()),
            timeout: Duration::from_millis(100),
            ..LlmConfig::default()
        };
        assert!(config.is_valid());

        let service = LlmService::new(config);
        assert!(service.enabled);
        // Will timeout since the URL is unreachable
        assert!(!service.check_sync("test", "ctx"));
    }
}

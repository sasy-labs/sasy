//! LLM Configuration
//!
//! Environment variable-based configuration for the LLM service.

use std::env;
use std::time::Duration;

/// LLM provider type
#[derive(Debug, Clone, PartialEq)]
pub enum LlmProvider {
    OpenAI,
    Azure,
}

impl LlmProvider {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Self::OpenAI),
            "azure" => Some(Self::Azure),
            _ => None,
        }
    }
}

/// LLM service configuration
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub enabled: bool,
    pub provider: LlmProvider,
    pub timeout: Duration,
    pub cache_ttl_secs: u64,

    /// Model name (used as OpenAI model or Azure deployment fallback).
    pub model: String,

    // OpenAI-specific
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,

    // Azure-specific
    pub azure_api_key: Option<String>,
    pub azure_endpoint: Option<String>,
    pub azure_deployment: Option<String>,
    pub azure_api_version: String,

    // Cache
    pub valkey_url: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: LlmProvider::OpenAI,
            timeout: Duration::from_secs(30),
            cache_ttl_secs: 3600, // 1 hour
            model: "gpt-4o-mini".to_string(),
            openai_api_key: None,
            openai_base_url: None,
            azure_api_key: None,
            azure_endpoint: None,
            azure_deployment: None,
            azure_api_version: "2024-02-01".to_string(),
            valkey_url: "redis://localhost:6379".to_string(),
        }
    }
}

/// Convert empty strings to None
fn non_empty_env(key: &str) -> Option<String> {
    env::var(key).ok().filter(|s| !s.is_empty())
}

impl LlmConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let enabled = env::var("LLM_ENABLED")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);

        let provider = env::var("LLM_PROVIDER")
            .ok()
            .and_then(|p| LlmProvider::from_str(&p))
            .unwrap_or(LlmProvider::OpenAI);

        let timeout_secs = env::var("LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        let cache_ttl_secs = env::var("LLM_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);

        Self {
            enabled,
            provider,
            timeout: Duration::from_secs(timeout_secs),
            cache_ttl_secs,
            model: env::var("LLM_CHECK_MODEL")
                .or_else(|_| env::var("OPENAI_MODEL"))
                .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            openai_api_key: non_empty_env("OPENAI_API_KEY"),
            openai_base_url: non_empty_env("OPENAI_BASE_URL"),
            azure_api_key: non_empty_env("AZURE_OPENAI_API_KEY")
                .or_else(|| non_empty_env("AZURE_API_KEY")),
            azure_endpoint: non_empty_env("AZURE_OPENAI_API_BASE")
                .or_else(|| non_empty_env("AZURE_OPENAI_ENDPOINT"))
                .or_else(|| non_empty_env("AZURE_ENDPOINT")),
            azure_deployment: non_empty_env("AZURE_OPENAI_DEPLOYMENT"),
            azure_api_version: env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2024-02-01".to_string()),
            valkey_url: env::var("VALKEY_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
        }
    }

    /// Check if the configuration is valid for the selected provider
    pub fn is_valid(&self) -> bool {
        if !self.enabled {
            return true; // Disabled is always valid
        }

        match self.provider {
            // OpenAI provider allows empty API key for compatible services
            LlmProvider::OpenAI => true,
            // Azure requires endpoint; deployment falls back to model name
            LlmProvider::Azure => self.azure_api_key.is_some() && self.azure_endpoint.is_some(),
        }
    }

    /// Get the Azure deployment name, falling back to model name if not set
    pub fn azure_deployment_or_model(&self) -> &str {
        self.azure_deployment.as_deref().unwrap_or(&self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.provider, LlmProvider::OpenAI);
        assert_eq!(config.model, "gpt-4o-mini");
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!(LlmProvider::from_str("openai"), Some(LlmProvider::OpenAI));
        assert_eq!(LlmProvider::from_str("OPENAI"), Some(LlmProvider::OpenAI));
        assert_eq!(LlmProvider::from_str("azure"), Some(LlmProvider::Azure));
        assert_eq!(LlmProvider::from_str("Azure"), Some(LlmProvider::Azure));
        assert_eq!(LlmProvider::from_str("invalid"), None);
    }

    #[test]
    fn test_disabled_is_always_valid() {
        let config = LlmConfig::default();
        assert!(config.is_valid());
    }
}

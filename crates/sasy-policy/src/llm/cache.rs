//! LLM Response Cache
//!
//! Caches LLM responses in Valkey (Redis-compatible) to avoid redundant API calls.

use redis::AsyncCommands;
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

/// Cache for LLM responses
pub struct LlmCache {
    client: redis::Client,
    ttl_secs: u64,
}

impl LlmCache {
    /// Create a new cache instance
    pub fn new(valkey_url: &str, ttl_secs: u64) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(valkey_url)?;
        Ok(Self { client, ttl_secs })
    }

    /// Generate a cache key from prompt and context
    pub fn cache_key(prompt: &str, context: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hasher.update(b"|");
        hasher.update(context.as_bytes());
        let hash = hasher.finalize();
        format!("llm_check:{:x}", hash)
    }

    /// Get a cached result
    pub async fn get(&self, prompt: &str, context: &str) -> Option<bool> {
        let key = Self::cache_key(prompt, context);

        match self.client.get_multiplexed_async_connection().await {
            Ok(mut conn) => match conn.get::<_, Option<String>>(&key).await {
                Ok(Some(value)) => {
                    debug!("Cache hit for key: {}", key);
                    Some(value == "true")
                }
                Ok(None) => {
                    debug!("Cache miss for key: {}", key);
                    None
                }
                Err(e) => {
                    warn!("Cache get error: {}", e);
                    None
                }
            },
            Err(e) => {
                warn!("Failed to connect to cache: {}", e);
                None
            }
        }
    }

    /// Store a result in cache
    pub async fn set(&self, prompt: &str, context: &str, result: bool) {
        let key = Self::cache_key(prompt, context);
        let value = if result { "true" } else { "false" };

        match self.client.get_multiplexed_async_connection().await {
            Ok(mut conn) => {
                if let Err(e) = conn.set_ex::<_, _, ()>(&key, value, self.ttl_secs).await {
                    warn!("Cache set error: {}", e);
                }
            }
            Err(e) => {
                warn!("Failed to connect to cache for set: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let key1 = LlmCache::cache_key("prompt1", "context1");
        let key2 = LlmCache::cache_key("prompt1", "context1");
        let key3 = LlmCache::cache_key("prompt2", "context1");

        assert!(key1.starts_with("llm_check:"));
        assert_eq!(key1, key2); // Same inputs -> same key
        assert_ne!(key1, key3); // Different inputs -> different key
    }

    #[test]
    fn test_cache_key_deterministic() {
        let key = LlmCache::cache_key("Does this contain PII?", "test data");
        assert_eq!(key.len(), 74); // "llm_check:" (10) + 64 hex chars
    }
}

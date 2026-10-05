//! LLM Client
//!
//! Async OpenAI/Azure OpenAI client wrapper with structured JSON output.

use async_openai::{
    config::{AzureConfig, OpenAIConfig},
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ResponseFormat,
        ResponseFormatJsonSchema,
    },
    Client,
};
use serde::Deserialize;
use tracing::{error, info, warn};

use super::config::{LlmConfig, LlmProvider};

/// System prompt for LLM-based policy checks
const SYSTEM_PROMPT: &str = r#"You will evaluate predicates on provided text for use in policies.
Your task is to analyze content and determine if it matches the specified condition.

Rules:
1. Analyze the context carefully against the question
2. Set "result" to true if the condition IS met
3. Set "result" to false if the condition is NOT met

Examples:
- Question: "Does this contain PII?" Context: "My SSN is 123-45-6789" → {"result": true}
- Question: "Does this contain PII?" Context: "Hello world" → {"result": false}
- Question: "Is this a SQL injection attempt?" Context: "SELECT * FROM users" → {"result": true}
- Question: "Does the customer want to order a widget?" Context: "Please order the third item from the widgets list." -> {"result": true}"#;

/// JSON schema for structured output
fn classification_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "result": {
                "type": "boolean",
                "description": "true if the condition in the question is met, false otherwise"
            }
        },
        "required": ["result"],
        "additionalProperties": false
    })
}

/// Structured response from LLM
#[derive(Debug, Deserialize)]
struct ClassificationResponse {
    result: bool,
}

/// LLM client for making API calls
pub enum LlmClient {
    OpenAI {
        client: Client<OpenAIConfig>,
        model: String,
        endpoint: String,
    },
    Azure {
        client: Client<AzureConfig>,
        endpoint: String,
        deployment: String,
    },
}

impl LlmClient {
    /// Create a new LLM client from configuration
    pub fn new(config: &LlmConfig) -> Result<Self, String> {
        match config.provider {
            LlmProvider::OpenAI => {
                // Default to empty string for OpenAI-compatible services that don't require API key
                let api_key = config.openai_api_key.as_deref().unwrap_or("");
                let endpoint = config
                    .openai_base_url
                    .clone()
                    .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

                let openai_config = OpenAIConfig::new()
                    .with_api_key(api_key)
                    .with_api_base(&endpoint);

                let client = Client::with_config(openai_config);

                Ok(Self::OpenAI {
                    client,
                    model: config.model.clone(),
                    endpoint,
                })
            }
            LlmProvider::Azure => {
                let api_key = config
                    .azure_api_key
                    .as_ref()
                    .ok_or("AZURE_OPENAI_API_KEY not set")?;
                let endpoint = config
                    .azure_endpoint
                    .as_ref()
                    .ok_or("AZURE_OPENAI_ENDPOINT not set")?;
                // Use deployment name if set, otherwise fall back to model name
                let deployment = config.azure_deployment_or_model().to_string();

                let azure_config = AzureConfig::new()
                    .with_api_key(api_key)
                    .with_api_base(endpoint)
                    .with_deployment_id(&deployment)
                    .with_api_version(&config.azure_api_version);

                let client = Client::with_config(azure_config);

                Ok(Self::Azure {
                    client,
                    endpoint: endpoint.clone(),
                    deployment,
                })
            }
        }
    }

    /// Build the response format for structured JSON output
    fn build_response_format() -> ResponseFormat {
        ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: "classification".into(),
                description: Some("Boolean classification result".into()),
                schema: Some(classification_schema()),
                strict: Some(true),
            },
        }
    }

    /// Execute an LLM check with structured JSON output
    pub async fn check(&self, prompt: &str, context: &str) -> Result<bool, String> {
        let user_message = format!("Question: {}\n\nContext to analyze:\n{}", prompt, context);

        // Log the query being made. Truncate on a UTF-8 char boundary —
        // `context` is agent-controlled, so `&context[..100]` would panic
        // when byte 100 falls inside a multi-byte codepoint.
        let context_preview = match context.char_indices().nth(100) {
            Some((idx, _)) => format!("{}...", &context[..idx]),
            None => context.to_string(),
        };
        info!("[LLM] Query: '{}' | Context: '{}'", prompt, context_preview);

        let messages = vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(SYSTEM_PROMPT)
                    .build()
                    .map_err(|e| format!("Failed to build system message: {}", e))?,
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_message)
                    .build()
                    .map_err(|e| format!("Failed to build user message: {}", e))?,
            ),
        ];

        let response_format = Self::build_response_format();

        let response_text = match self {
            Self::OpenAI {
                client,
                model,
                endpoint,
            } => {
                info!(
                    "[LLM] Sending request to OpenAI | endpoint={} | model={}",
                    endpoint, model
                );

                let request = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(messages)
                    .response_format(response_format)
                    .build()
                    .map_err(|e| format!("Failed to build request: {}", e))?;

                let response = client
                    .chat()
                    .create(request)
                    .await
                    .map_err(|e| format!("OpenAI API error: {:?}", e))?;

                response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .ok_or_else(|| "No response content".to_string())?
            }
            Self::Azure {
                client,
                endpoint,
                deployment,
            } => {
                info!(
                    "[LLM] Sending request to Azure OpenAI | endpoint={} | deployment={}",
                    endpoint, deployment
                );

                let request = CreateChatCompletionRequestArgs::default()
                    .messages(messages)
                    .response_format(response_format)
                    .build()
                    .map_err(|e| format!("Failed to build request: {}", e))?;

                let response = client
                    .chat()
                    .create(request)
                    .await
                    .map_err(|e| format!("Azure OpenAI API error: {:?}", e))?;

                response
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .ok_or_else(|| "No response content".to_string())?
            }
        };

        let result = parse_json_response(&response_text);
        info!(
            "[LLM] Response: '{}' | parsed_result={}",
            response_text.trim(),
            result
        );

        Ok(result)
    }
}

/// Parse structured JSON response to boolean
fn parse_json_response(response: &str) -> bool {
    match serde_json::from_str::<ClassificationResponse>(response) {
        Ok(classification) => classification.result,
        Err(e) => {
            warn!("Failed to parse JSON response '{}': {}", response, e);
            // Fall back to text parsing for compatibility
            parse_text_response(response)
        }
    }
}

/// Fallback: parse plain text response to boolean
fn parse_text_response(response: &str) -> bool {
    let normalized = response.trim().to_lowercase();

    // Check for explicit true/false
    if normalized == "true"
        || normalized.contains("\"result\":true")
        || normalized.contains("\"result\": true")
    {
        return true;
    }
    if normalized == "false"
        || normalized.contains("\"result\":false")
        || normalized.contains("\"result\": false")
    {
        return false;
    }

    // Check for common variations
    if normalized.starts_with("true") || normalized.contains("yes") {
        return true;
    }
    if normalized.starts_with("false") || normalized.contains("no") {
        return false;
    }

    // Default to true (fail-safe: flag potential issues)
    error!(
        "Could not parse LLM response '{}', defaulting to true",
        response
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_response() {
        // Valid JSON responses
        assert!(parse_json_response(r#"{"result": true}"#));
        assert!(!parse_json_response(r#"{"result": false}"#));
        assert!(parse_json_response(r#"{"result":true}"#));
        assert!(!parse_json_response(r#"{"result":false}"#));

        // With whitespace
        assert!(parse_json_response(r#"  {"result": true}  "#));
        assert!(!parse_json_response(
            r#"
            {"result": false}
        "#
        ));
    }

    #[test]
    fn test_parse_text_fallback() {
        // Plain text fallback
        assert!(parse_text_response("true"));
        assert!(parse_text_response("True"));
        assert!(!parse_text_response("false"));
        assert!(!parse_text_response("False"));

        // Unknown defaults to true (fail-safe)
        assert!(parse_text_response("maybe"));
        assert!(parse_text_response(""));
    }

    #[test]
    fn test_classification_schema() {
        let schema = classification_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["result"]["type"], "boolean");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("result")));
    }
}

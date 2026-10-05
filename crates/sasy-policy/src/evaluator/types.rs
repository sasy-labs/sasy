//! Types for the evaluator interface.
//!
//! Backend-agnostic types used across the evaluator trait and IPC wire
//! protocol. They mirror the authorization types but are decoupled from
//! backend-specific representations.

use serde::{Deserialize, Serialize};

use crate::engine::GraphUpdate;

/// Authorization query sent to the evaluator.
///
/// Instance relations (Current, Actions, entity, roles) are query
/// parameters, not persistent state.
///
/// Note: the wire `tenant_id` field on the proto is intentionally
/// not surfaced here — server code derives the tenant from auth
/// (see `sasy_auth::request_tenant`) and combines it with the
/// session_id field as a [`sasy_common::SessionScope`] before
/// dispatch. This keeps cross-tenant traffic from forging a
/// `tenant_id` query parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalAuthRequest {
    pub current_node_ids: Vec<String>,
    pub actions: Vec<EvalAction>,
    pub entity: Option<String>,
    pub roles: Vec<String>,
    /// Auth-derived tenant. Carried for diagnostic/log use; the
    /// actual partition key is the [`sasy_common::SessionScope`]
    /// constructed at dispatch.
    pub tenant_id: Option<String>,
    /// Session/conversation partition (within the request tenant).
    /// The worker scopes its evaluator state to this session's
    /// messages before running the query. ``None`` or empty string
    /// means the per-tenant global partition.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Server-stamped principal (auth-derived). Distinct from
    /// `entity` (the user-supplied actor); this is the immutable
    /// identity attribute. `None` for anonymous requests.
    #[serde(default)]
    pub principal: Option<String>,
    /// Per-action metadata facts → `ActionMetadata(idx, rel, a, b)` EDB,
    /// indexed by position in `actions`. Daemon-resolved external context
    /// for a specific action (e.g. supply-chain verdicts). Empty for most
    /// requests; only actions the caller enriched carry an entry.
    #[serde(default)]
    pub action_metadata: Vec<ActionMetadataEntry>,
}

/// Metadata facts attached to one action (by `index` into the request's
/// `actions`) → `ActionMetadata(index, rel, a, b)` tuples for the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionMetadataEntry {
    pub index: u32,
    pub facts: Vec<PolicyMetadataFact>,
}

/// Action in evaluator-native format (serializable over IPC).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalAction {
    HttpRequest {
        url: String,
        body: String,
        headers: Vec<(String, String)>,
    },
    ToolCall {
        fn_name: String,
        args: String,
    },
    SendMessage {
        content: String,
        agent: String,
        agent_role: String,
        tool_calls: Vec<(String, String)>,
        entity: Option<String>,
    },
}

/// Authorization result from the evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalAuthResponse {
    pub results: Vec<EvalActionResult>,
}

/// One static policy-metadata fact → `PolicyMetadata(rel, a, b)` EDB tuple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetadataFact {
    pub rel: String,
    pub a: String,
    pub b: String,
}

/// A denial reason produced by a fired DenialReason rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDenialReason {
    /// "block" (hard deny) or "ask" (soft, requires user approval).
    #[serde(default = "default_kind")]
    pub kind: String,
    pub reason: String,
    pub suggestion: String,
}

fn default_kind() -> String {
    "block".to_string()
}

/// Diagnostic-only row; never contributes to the evaluator's decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalAllowRoute {
    pub rule_id: String,
    pub status: String,
    pub details: String,
    pub suggestion: String,
    pub source_location: String,
}

fn deserialize_allow_routes<'de, D>(deserializer: D) -> Result<Vec<EvalAllowRoute>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // The enclosing IPC frame/JSON must still be valid. Optional diagnostic
    // data from an older or third-party evaluator must not invalidate its
    // otherwise well-formed decision.
    let value = serde_json::Value::deserialize(deserializer)?;
    let Some(rows) = value.as_array() else {
        return Ok(Vec::new());
    };
    if rows.len() > 512 {
        return Ok(Vec::new());
    }
    let mut remaining = 262_144usize;
    for row in rows {
        for name in [
            "rule_id",
            "status",
            "details",
            "suggestion",
            "source_location",
        ] {
            let Some(text) = row.get(name).and_then(serde_json::Value::as_str) else {
                return Ok(Vec::new());
            };
            let Some(next) = remaining.checked_sub(text.len()) else {
                return Ok(Vec::new());
            };
            remaining = next;
        }
    }
    Ok(serde_json::from_value(value).unwrap_or_default())
}

/// Result for a single action from the evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalActionResult {
    pub index: u32,
    pub authorized: bool,
    pub is_authenticated: bool,
    pub is_denylisted: bool,
    pub is_allowlisted: bool,
    pub transform_ids: Vec<String>,
    pub deny_if_unauthorized: bool,
    pub allow_passthrough: bool,
    /// Soft denial: an "ask"-kind reason is present and no block reason —
    /// the host should prompt the user (deny > ask > allow).
    #[serde(default)]
    pub requires_approval: bool,
    /// Denial reasons from DenialReason tuples (correct by construction).
    #[serde(default)]
    pub denial_reasons: Vec<EvalDenialReason>,
    /// Empty for old evaluators or unsupported source; retain legacy hints.
    #[serde(
        default,
        deserialize_with = "deserialize_allow_routes",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allow_routes: Vec<EvalAllowRoute>,
}

#[cfg(test)]
mod attribution_wire_tests {
    use super::*;

    #[test]
    fn absent_diagnostics_preserve_legacy_result_round_trip() {
        let legacy = serde_json::json!({"index":0,"authorized":false,
            "is_authenticated":true,"is_denylisted":true,"is_allowlisted":false,
            "transform_ids":[],"deny_if_unauthorized":true,"allow_passthrough":false,
            "requires_approval":false,"denial_reasons":[]});
        let result: EvalActionResult = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(serde_json::to_value(result).unwrap(), legacy);
    }

    #[test]
    fn malformed_optional_diagnostics_preserve_valid_decision() {
        let base = serde_json::json!({"index":0,"authorized":false,
            "is_authenticated":true,"is_denylisted":true,"is_allowlisted":false,
            "transform_ids":[],"deny_if_unauthorized":true,"allow_passthrough":false});
        for optional in [
            serde_json::Value::Null,
            serde_json::json!({}),
            serde_json::json!([{"rule_id":"incomplete"}]),
            serde_json::json!([1, 2, 3]),
        ] {
            let mut input = base.clone();
            input["allow_routes"] = optional;
            let result: EvalActionResult = serde_json::from_value(input).unwrap();
            assert!(result.is_denylisted && !result.authorized);
            assert!(result.allow_routes.is_empty());
        }
    }
}

/// IPC message from policy engine to evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalRequest {
    Update {
        updates: Vec<GraphUpdate>,
    },
    /// Set the static policy-metadata facts (PolicyMetadata EDB). Sent once
    /// after the evaluator spawns; constant for its life. Decoupled from the
    /// compiled policy so a precompiled (restricted-build) evaluator accepts
    /// config without recompiling. JSON: {"SetMetadata":{"facts":[{rel,a,b}]}}.
    SetMetadata {
        facts: Vec<PolicyMetadataFact>,
    },
    Query(EvalAuthRequest),
    Reset,
    /// Response to an LlmQuery oracle callback from the evaluator.
    LlmResult {
        result: bool,
    },
    Shutdown,
}

/// IPC message from evaluator to policy engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalResponse {
    UpdateOk,
    QueryResult(EvalAuthResponse),
    ResetOk,
    Error {
        message: String,
    },
    /// Oracle callback: the C++ functor needs an LLM check result.
    /// Sent during a Query when @llm_check_fn hits a cache miss.
    LlmQuery {
        prompt: String,
        context: String,
    },
}

/// Convert a proto Action to an EvalAction.
impl EvalAction {
    pub fn from_proto(action: &sasy_common::policy_engine::Action) -> Option<Self> {
        use sasy_common::policy_engine::action::ActionType;
        match &action.action_type {
            Some(ActionType::HttpRequest(req)) => Some(EvalAction::HttpRequest {
                url: req.url.clone(),
                body: req.body.clone(),
                headers: req
                    .headers
                    .iter()
                    .map(|h| (h.key.clone(), h.value.clone()))
                    .collect(),
            }),
            Some(ActionType::ToolCall(tc)) => Some(EvalAction::ToolCall {
                fn_name: tc.fn_name.clone(),
                args: tc.args.clone(),
            }),
            Some(ActionType::SendMessage(msg)) => Some(EvalAction::SendMessage {
                content: msg.content.clone(),
                agent: msg.agent.clone(),
                agent_role: msg.agent_role.clone(),
                tool_calls: msg
                    .tool_calls
                    .iter()
                    .map(|tc| (tc.fn_name.clone(), tc.args.clone()))
                    .collect(),
                entity: msg.entity.clone(),
            }),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_query_response_round_trip() {
        let resp = EvalResponse::LlmQuery {
            prompt: "Does this contain PII?".into(),
            context: "My SSN is 123-45-6789".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deser: EvalResponse = serde_json::from_str(&json).unwrap();
        match deser {
            EvalResponse::LlmQuery { prompt, context } => {
                assert_eq!(prompt, "Does this contain PII?");
                assert_eq!(context, "My SSN is 123-45-6789");
            }
            other => panic!("Expected LlmQuery, got {:?}", other),
        }
    }

    #[test]
    fn llm_result_request_round_trip() {
        for expected in [true, false] {
            let req = EvalRequest::LlmResult { result: expected };
            let json = serde_json::to_string(&req).unwrap();
            let deser: EvalRequest = serde_json::from_str(&json).unwrap();
            match deser {
                EvalRequest::LlmResult { result } => assert_eq!(result, expected),
                other => panic!("Expected LlmResult, got {:?}", other),
            }
        }
    }

    #[test]
    fn llm_query_json_matches_cpp_format() {
        // The C++ shim parses response["LlmQuery"]["prompt"] etc.
        let resp = EvalResponse::LlmQuery {
            prompt: "test".into(),
            context: "ctx".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["LlmQuery"]["prompt"], "test");
        assert_eq!(parsed["LlmQuery"]["context"], "ctx");
    }

    #[test]
    fn llm_result_json_matches_cpp_format() {
        // `LlmResult` is an `EvalRequest` (Rust→C++); the shim reads it
        // as request["LlmResult"]["result"].
        let req = EvalRequest::LlmResult { result: true };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["LlmResult"]["result"], true);
    }

    #[test]
    fn unknown_request_variant_is_rejected() {
        let unknown = r#"{"SetLlmCache":{"entries":[]}}"#;
        let result = serde_json::from_str::<EvalRequest>(unknown);
        assert!(result.is_err(), "an unknown variant must not deserialize");
    }
}

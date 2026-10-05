//! Denial trace generation for policy decisions.

use serde::{Deserialize, Serialize};

use crate::policy_types::AuthAction;
use crate::rule_metadata::RuleMetadata;

/// Validate and group optional evaluator rows without changing a decision.
/// A malformed, conflicting, or over-budget set falls back to legacy hints.
pub fn group_allow_routes(
    rows: &[crate::evaluator::types::EvalAllowRoute],
) -> Vec<sasy_common::policy_engine::AllowRouteDiagnostic> {
    use sasy_common::policy_engine::AllowRouteDiagnostic;
    use std::collections::BTreeMap;
    if rows.len() > 512 {
        return Vec::new();
    }
    let mut remaining = 262_144usize;
    let mut groups: BTreeMap<String, AllowRouteDiagnostic> = BTreeMap::new();
    for row in rows {
        if row.rule_id.is_empty()
            || !matches!(row.status.as_str(), "blocked" | "possible" | "unknown")
        {
            return Vec::new();
        }
        for text in [
            &row.rule_id,
            &row.status,
            &row.details,
            &row.suggestion,
            &row.source_location,
        ] {
            let Some(next) = remaining.checked_sub(text.len()) else {
                return Vec::new();
            };
            remaining = next;
        }
        let source_location =
            (!row.source_location.is_empty()).then(|| row.source_location.clone());
        let group = groups
            .entry(row.rule_id.clone())
            .or_insert_with(|| AllowRouteDiagnostic {
                rule_id: row.rule_id.clone(),
                status: row.status.clone(),
                details: row.details.clone(),
                suggestions: Vec::new(),
                source_location: source_location.clone(),
            });
        if group.status != row.status
            || group.details != row.details
            || group.source_location != source_location
        {
            return Vec::new();
        }
        // An authored suggestion on a route ruled out by fixed context is not
        // a repair for this request. Retain the blocker, suppress that hint.
        if row.status != "blocked"
            && !row.suggestion.is_empty()
            && !group.suggestions.contains(&row.suggestion)
        {
            group.suggestions.push(row.suggestion.clone());
        }
        if groups.len() > 64 {
            return Vec::new();
        }
    }
    groups.into_values().collect()
}

/// Denial trace explaining why an action was denied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialTrace {
    /// Description of the action that was denied.
    pub action_description: String,
    /// Reasons for denial.
    pub reasons: Vec<DenialReason>,
    /// Suggested fixes.
    pub suggested_fixes: Vec<String>,
}

/// A single reason for denial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DenialReason {
    NotAuthenticated {
        #[serde(skip_serializing_if = "Option::is_none")]
        source_location: Option<String>,
    },
    Denylisted {
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_location: Option<String>,
    },
    NotAllowlisted {
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_location: Option<String>,
    },
    SyncTimeout {
        pending_nodes: Vec<String>,
    },
    /// Soft denial: the policy requests user approval (`@ask`). A host that can
    /// prompt asks the user; otherwise this degrades to a deny.
    Ask {
        message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_location: Option<String>,
    },
}

impl std::fmt::Display for DenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenialReason::NotAuthenticated { .. } => {
                write!(f, "Not authenticated")
            }
            DenialReason::Denylisted { message, .. } => {
                if let Some(msg) = message {
                    write!(f, "Denylisted: {}", msg)
                } else {
                    write!(f, "Action is denylisted")
                }
            }
            DenialReason::Ask { message, .. } => {
                if let Some(msg) = message {
                    write!(f, "Approval required: {}", msg)
                } else {
                    write!(f, "Approval required")
                }
            }
            DenialReason::NotAllowlisted { message, .. } => {
                if let Some(msg) = message {
                    write!(f, "Not allowlisted: {}", msg)
                } else {
                    write!(f, "Action not on allowlist")
                }
            }
            DenialReason::SyncTimeout { pending_nodes } => {
                write!(f, "Sync timeout: {} node(s) pending", pending_nodes.len())
            }
        }
    }
}

/// Builder for creating denial traces.
pub struct TraceBuilder {
    action_description: String,
    reasons: Vec<DenialReason>,
    suggested_fixes: Vec<String>,
}

impl TraceBuilder {
    pub fn new(action: &AuthAction) -> Self {
        let action_description = match action {
            AuthAction::HttpRequest { url, .. } => {
                format!("HTTP request to {}", url)
            }
            AuthAction::ToolCall { fn_name, .. } => {
                format!("Tool call: {}", fn_name)
            }
            AuthAction::SendMessage { agent, .. } => {
                format!("Send message from agent '{}'", agent)
            }
        };

        Self {
            action_description,
            reasons: Vec::new(),
            suggested_fixes: Vec::new(),
        }
    }

    /// Apply suggestions from parsed policy annotations.
    pub fn with_rule_metadata(mut self, metadata: &RuleMetadata) -> Self {
        for suggestion in &metadata.suggestions {
            self.suggested_fixes.push(suggestion.clone());
        }
        self
    }

    pub fn not_authenticated(mut self, source_location: Option<String>) -> Self {
        self.reasons
            .push(DenialReason::NotAuthenticated { source_location });
        self.suggested_fixes
            .push("Ensure request includes authenticated entity".to_string());
        self
    }

    pub fn denylisted(mut self, message: Option<String>, source_location: Option<String>) -> Self {
        self.reasons.push(DenialReason::Denylisted {
            message,
            source_location,
        });
        self
    }

    pub fn ask(mut self, message: Option<String>, source_location: Option<String>) -> Self {
        self.reasons.push(DenialReason::Ask {
            message,
            source_location,
        });
        self
    }

    pub fn not_allowlisted(
        mut self,
        message: Option<String>,
        source_location: Option<String>,
    ) -> Self {
        self.reasons.push(DenialReason::NotAllowlisted {
            message,
            source_location,
        });
        self
    }

    pub fn sync_timeout(mut self, pending_nodes: Vec<String>) -> Self {
        self.reasons
            .push(DenialReason::SyncTimeout { pending_nodes });
        self
    }

    pub fn suggest(mut self, fix: &str) -> Self {
        self.suggested_fixes.push(fix.to_string());
        self
    }

    /// Get the current count of suggested fixes.
    pub fn suggested_fixes_count(&self) -> usize {
        self.suggested_fixes.len()
    }

    pub fn build(self) -> DenialTrace {
        DenialTrace {
            action_description: self.action_description,
            reasons: self.reasons,
            suggested_fixes: self.suggested_fixes,
        }
    }
}

/// Generate contextual suggestions based on the action type.
///
/// Policy-specific suggestions should be defined via
/// @suggestion annotations in the policy file. This
/// function provides generic fallback suggestions only.
pub fn generate_contextual_suggestions(action: &AuthAction) -> Vec<String> {
    let mut suggestions = Vec::new();

    match action {
        AuthAction::HttpRequest { .. } => {
            suggestions.push("Check policy rules for this API endpoint".to_string());
        }
        AuthAction::ToolCall { fn_name, .. } => {
            suggestions.push(format!(
                "Ensure tool '{}' is registered in the \
                 policy allowlist",
                fn_name
            ));
        }
        AuthAction::SendMessage { agent, .. } => {
            suggestions.push(format!(
                "Verify agent '{}' has permission to \
                 send messages",
                agent
            ));
        }
    }

    suggestions
}

/// Check if a pattern matches a value using regex.
fn pattern_matches(pattern: &str, value: &str) -> bool {
    match regex::Regex::new(pattern) {
        Ok(re) => re.is_match(value),
        Err(e) => {
            tracing::warn!("Invalid regex pattern '{}': {}", pattern, e);
            value.contains(pattern)
        }
    }
}

/// Check if a rule's pattern annotations match the action.
pub fn rule_matches_action(
    action: &AuthAction,
    custom_annotations: &std::collections::HashMap<String, String>,
) -> bool {
    match action {
        AuthAction::HttpRequest { url, .. } => {
            if let Some(url_pattern) = custom_annotations.get("url_pattern") {
                pattern_matches(url_pattern, url)
            } else {
                !custom_annotations.contains_key("tool_pattern")
            }
        }
        AuthAction::ToolCall { fn_name, .. } => {
            if let Some(tool_pattern) = custom_annotations.get("tool_pattern") {
                pattern_matches(tool_pattern, fn_name)
            } else {
                !custom_annotations.contains_key("url_pattern")
            }
        }
        AuthAction::SendMessage { .. } => {
            !custom_annotations.contains_key("url_pattern")
                && !custom_annotations.contains_key("tool_pattern")
        }
    }
}

/// Convert internal [`DenialReason`] to proto `DenialReason`.
pub fn denial_reason_to_proto(reason: &DenialReason) -> sasy_common::policy_engine::DenialReason {
    match reason {
        DenialReason::NotAuthenticated { source_location } => {
            sasy_common::policy_engine::DenialReason {
                reason_type: sasy_common::policy_engine::DenialReasonType::NotAuthenticated as i32,
                details: "Not authenticated".into(),
                source_location: source_location.clone(),
            }
        }
        DenialReason::Denylisted {
            message,
            source_location,
        } => sasy_common::policy_engine::DenialReason {
            reason_type: sasy_common::policy_engine::DenialReasonType::Denylisted as i32,
            details: message
                .clone()
                .unwrap_or_else(|| "Action is denylisted".into()),
            source_location: source_location.clone(),
        },
        DenialReason::NotAllowlisted {
            message,
            source_location,
        } => sasy_common::policy_engine::DenialReason {
            reason_type: sasy_common::policy_engine::DenialReasonType::NotAllowlisted as i32,
            details: message
                .clone()
                .unwrap_or_else(|| "Action not on allowlist".into()),
            source_location: source_location.clone(),
        },
        DenialReason::SyncTimeout { pending_nodes } => sasy_common::policy_engine::DenialReason {
            reason_type: sasy_common::policy_engine::DenialReasonType::SyncTimeout as i32,
            details: format!("Sync timeout: {} node(s) pending", pending_nodes.len()),
            source_location: None,
        },
        DenialReason::Ask {
            message,
            source_location,
        } => sasy_common::policy_engine::DenialReason {
            reason_type: sasy_common::policy_engine::DenialReasonType::Ask as i32,
            details: message
                .clone()
                .unwrap_or_else(|| "Approval required".into()),
            source_location: source_location.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_types::AuthAction;

    fn route(id: &str, status: &str, suggestion: &str) -> crate::evaluator::types::EvalAllowRoute {
        crate::evaluator::types::EvalAllowRoute {
            rule_id: id.into(),
            status: status.into(),
            details: "fixed condition projection".into(),
            suggestion: suggestion.into(),
            source_location: "policy:12".into(),
        }
    }

    #[test]
    fn allow_routes_preserve_alternatives_and_suppress_blocked_hints() {
        let rows = vec![
            route("a", "blocked", "wrong principal's approval"),
            route("b", "possible", "request approval"),
            route("b", "possible", "request approval"),
            route("c", "unknown", "author hint"),
        ];
        let groups = group_allow_routes(&rows);
        assert_eq!(groups.len(), 3);
        assert!(groups[0].suggestions.is_empty());
        assert_eq!(groups[1].suggestions, ["request approval"]);
        assert_eq!(groups[2].status, "unknown");
    }

    #[test]
    fn allow_routes_discard_conflicting_or_over_budget_attribution() {
        assert!(
            group_allow_routes(&[route("a", "possible", "x"), route("a", "blocked", "x")])
                .is_empty()
        );
        assert!(group_allow_routes(&[route("a", "unrecognized", "x")]).is_empty());
        assert!(group_allow_routes(&vec![route("a", "possible", "x"); 513]).is_empty());
        assert!(group_allow_routes(&[route("a", "possible", &"x".repeat(262_145))]).is_empty());
    }

    #[test]
    fn sync_timeout_display() {
        let reason = DenialReason::SyncTimeout {
            pending_nodes: vec!["n1".into(), "n2".into()],
        };
        assert_eq!(reason.to_string(), "Sync timeout: 2 node(s) pending");
    }

    #[test]
    fn trace_builder_sync_timeout() {
        let action = AuthAction::HttpRequest {
            url: "https://api.example.com".into(),
            body: "{}".into(),
            headers: vec![],
        };
        let trace = TraceBuilder::new(&action)
            .sync_timeout(vec!["n1".into()])
            .suggest("Retry the request")
            .build();

        assert_eq!(trace.reasons.len(), 1);
        assert!(matches!(
            &trace.reasons[0],
            DenialReason::SyncTimeout {
                pending_nodes,
            } if pending_nodes == &["n1"]
        ));
        assert_eq!(trace.suggested_fixes.len(), 1);
    }

    #[test]
    fn sync_timeout_serialization_roundtrip() {
        let reason = DenialReason::SyncTimeout {
            pending_nodes: vec!["n1".into()],
        };
        let json = serde_json::to_string(&reason).unwrap();
        let deserialized: DenialReason = serde_json::from_str(&json).unwrap();
        assert_eq!(reason.to_string(), deserialized.to_string());
    }
}

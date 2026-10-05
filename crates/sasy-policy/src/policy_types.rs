//! Backend-neutral policy domain types shared by the Soufflé evaluator and
//! gRPC service layer. Each backend converts them at its own boundary.

use serde::{Deserialize, Serialize};

/// An action to be authorized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthAction {
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

/// Convert a proto Action to an AuthAction, returning None if the action type is missing.
pub fn proto_action_to_auth_option(
    action: &sasy_common::policy_engine::Action,
) -> Option<AuthAction> {
    use sasy_common::policy_engine::action::ActionType;
    match &action.action_type {
        Some(ActionType::HttpRequest(req)) => Some(AuthAction::HttpRequest {
            url: req.url.clone(),
            body: req.body.clone(),
            headers: req
                .headers
                .iter()
                .map(|h| (h.key.clone(), h.value.clone()))
                .collect(),
        }),
        Some(ActionType::ToolCall(tc)) => Some(AuthAction::ToolCall {
            fn_name: tc.fn_name.clone(),
            args: tc.args.clone(),
        }),
        Some(ActionType::SendMessage(sm)) => Some(AuthAction::SendMessage {
            content: sm.content.clone(),
            agent: sm.agent.clone(),
            agent_role: sm.agent_role.clone(),
            tool_calls: sm
                .tool_calls
                .iter()
                .map(|tc| (tc.fn_name.clone(), tc.args.clone()))
                .collect(),
            entity: sm.entity.clone(),
        }),
        None => None,
    }
}

/// Authorization request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationRequest {
    pub current_node_ids: Vec<String>,
    pub actions: Vec<AuthAction>,
    pub entity: Option<String>,
    pub roles: Vec<String>,
}

/// Authorization response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationResponse {
    pub results: Vec<ActionResult>,
}

/// Result for a single action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub index: u32,
    pub authorized: bool,
    pub trace: Option<crate::trace::DenialTrace>,
    pub transform_ids: Vec<String>,
    pub deny_if_unauthorized: bool,
}

/// A tool call in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// A node in the observability graph.
///
/// Extended version of [`crate::engine::GraphUpdate`] with
/// the full node fields (tools, entity, derived_from) a policy
/// backend may need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub content: Option<String>,
    pub role: Option<String>,
    pub agent: Option<String>,
    pub tools: Vec<ToolCall>,
    pub entity: Option<String>,
    /// If this message is a tool result, the tool it came
    /// from.
    pub derived_from: Option<ToolCall>,
}

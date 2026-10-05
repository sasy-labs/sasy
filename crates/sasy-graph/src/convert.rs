//! Conversions between proto types and internal storage
//! types.

use sasy_common::observability::{Computation, Edge, Event, Tool};
use sasy_common::SessionScope;

use crate::types::{ComputationNode, EdgeData, EdgeKindSer, MessageNode, NodeData};

// ── Event <-> MessageNode ───────────────────────────

/// Convert a proto [`Event`] to an internal [`MessageNode`].
///
/// `scope` is the auth-derived `(tenant, session)` partition from
/// the request envelope (every event in a batch shares it).
/// `principal` is the auth-derived identity stamped onto the node.
/// Any value the client sent on `event.principal` is ignored.
pub fn event_to_message(
    event: &Event,
    id: &str,
    scope: SessionScope,
    principal: Option<&str>,
) -> MessageNode {
    // Canonical proto-Role -> lowercase string via the shared helper,
    // so this mapping can't drift from the others.
    let role = event
        .role
        .and_then(sasy_common::MessageRole::from_proto_opt)
        .map(|m| m.as_str().to_string());

    let tools_json = if event.tools.is_empty() {
        None
    } else {
        Some(tools_to_json(&event.tools))
    };

    let derived_from_json = event.derived_from.as_ref().map(tool_to_json);

    MessageNode {
        id: id.to_string(),
        content: event.text.clone(),
        role,
        agent: event.agent.clone(),
        tools_json,
        derived_from_json,
        scope,
        principal: principal.map(|s| s.to_string()),
        entity: event.entity.clone(),
        metadata: event.metadata.clone(),
    }
}

pub fn message_to_event(msg: &MessageNode) -> Event {
    let role = msg
        .role
        .as_deref()
        .and_then(sasy_common::MessageRole::from_str_opt)
        .map(sasy_common::MessageRole::to_proto);

    let tools = msg
        .tools_json
        .as_deref()
        .and_then(json_to_tools)
        .unwrap_or_default();

    let derived_from = msg.derived_from_json.as_deref().and_then(json_to_tool);

    Event {
        text: msg.content.clone(),
        agent: msg.agent.clone(),
        role,
        id: Some(msg.id.clone()),
        tools,
        derived_from,
        principal: msg.principal.clone(),
        entity: msg.entity.clone(),
        metadata: msg.metadata.clone(),
    }
}

// ── Computation <-> ComputationNode ─────────────────

pub fn proto_to_comp(
    c: &Computation,
    scope: SessionScope,
    principal: Option<&str>,
) -> ComputationNode {
    ComputationNode {
        span_id: c.span_id.clone(),
        trace_id: c.trace_id.clone(),
        parent_span_id: c.parent_span_id.clone(),
        name: c.name.clone(),
        start_time_ns: c.start_time_ns,
        end_time_ns: c.end_time_ns,
        duration_ns: c.duration_ns,
        status_code: c.status_code,
        status_message: c.status_message.clone(),
        attributes_json: c.attributes_json.clone(),
        events_json: c.events_json.clone(),
        service_name: c.service_name.clone(),
        service_version: c.service_version.clone(),
        input_message_ids: c.input_message_ids.clone(),
        output_message_id: c.output_message_id.clone(),
        scope,
        principal: principal.map(|s| s.to_string()),
        entity: c.entity.clone(),
    }
}

pub fn comp_to_proto(c: &ComputationNode) -> Computation {
    Computation {
        span_id: c.span_id.clone(),
        trace_id: c.trace_id.clone(),
        parent_span_id: c.parent_span_id.clone(),
        name: c.name.clone(),
        start_time_ns: c.start_time_ns,
        end_time_ns: c.end_time_ns,
        duration_ns: c.duration_ns,
        status_code: c.status_code,
        status_message: c.status_message.clone(),
        attributes_json: c.attributes_json.clone(),
        events_json: c.events_json.clone(),
        service_name: c.service_name.clone(),
        service_version: c.service_version.clone(),
        input_message_ids: c.input_message_ids.clone(),
        output_message_id: c.output_message_id.clone(),
        linked_span_ids: vec![],
        principal: c.principal.clone(),
        entity: c.entity.clone(),
    }
}

// ── Edge <-> EdgeData ───────────────────────────────

/// Pre-extract source/destination/index/proximal from a proto
/// [`Edge`]. The session is an envelope field: every edge in a
/// batch shares the envelope's scope.
pub fn proto_edge_to_data(e: &Edge) -> (String, String, Option<u32>, Option<bool>) {
    (
        e.source.clone(),
        e.destination.clone(),
        e.message_index,
        e.proximal,
    )
}

pub fn edge_data_to_proto(source: &str, destination: &str, data: &EdgeData) -> Edge {
    Edge {
        source: source.to_string(),
        destination: destination.to_string(),
        message_index: data.message_index,
        proximal: data.proximal,
        principal: data.principal.clone(),
        entity: data.entity.clone(),
    }
}

/// Convert only DEPENDS_ON edges to proto Edge.
pub fn depends_on_to_proto(source: &str, dest: &str, data: &EdgeData) -> Option<Edge> {
    if data.kind == EdgeKindSer::DependsOn {
        Some(edge_data_to_proto(source, dest, data))
    } else {
        None
    }
}

// ── NodeData -> Event (for slicing output) ──────────

pub fn node_to_event(node: &NodeData) -> Option<Event> {
    match node {
        NodeData::Message(m) => Some(message_to_event(m)),
        NodeData::Computation(_) => None,
    }
}

// ── Tool JSON helpers ───────────────────────────────

fn tools_to_json(tools: &[Tool]) -> String {
    let arr: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "arguments": t.arguments,
            })
        })
        .collect();
    serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
}

fn tool_to_json(tool: &Tool) -> String {
    serde_json::to_string(&serde_json::json!({
        "name": tool.name,
        "arguments": tool.arguments,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

fn json_to_tools(json: &str) -> Option<Vec<Tool>> {
    let arr: Vec<serde_json::Value> = serde_json::from_str(json).ok()?;
    Some(
        arr.iter()
            .map(|v| Tool {
                name: v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string()),
                arguments: v
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .map(|s| s.to_string()),
            })
            .collect(),
    )
}

fn json_to_tool(json: &str) -> Option<Tool> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    Some(Tool {
        name: v
            .get("name")
            .and_then(|n| n.as_str())
            .map(|s| s.to_string()),
        arguments: v
            .get("arguments")
            .and_then(|a| a.as_str())
            .map(|s| s.to_string()),
    })
}

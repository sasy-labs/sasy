//! Optional background sync from GraphStore to Neo4j.
//!
//! Subscribes to graph update broadcasts and replicates them to a
//! Neo4j instance asynchronously. This enables the hybrid architecture:
//! RocksDB for fast policy evaluation, Neo4j for analytics and the
//! browser UI.
//!
//! The mirror is **per binary, not per tenant**: one `sasy serve` writes
//! every tenant it hosts into the one Neo4j database named by its config.
//! Mirrored nodes therefore carry a `tenant` property and are keyed on
//! `(id, tenant)`, so two tenants that use the same message id stay two
//! nodes. Anything reading the mirror has to filter on `tenant` itself —
//! the mirror grants no isolation of its own.
//!
//! Usage:
//! ```ignore
//! let config = sasy_graph::neo4j_sync::Neo4jSyncConfig { /* ... */ };
//! let handle = sasy_graph::neo4j_sync::start(store.clone(), config);
//! ```

use std::sync::Arc;

use crate::GraphStore;
use tracing::{debug, error, info, warn};

/// Configuration for Neo4j background sync.
#[derive(Debug, Clone)]
pub struct Neo4jSyncConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
}

/// Checks a database name before the mirror places it in the transactional
/// endpoint path (`{uri}/db/{database}/tx/commit`).
///
/// The name comes from the operator (`--neo4j-database`), never from a caller,
/// but it is the one value the mirror still interpolates into a request rather
/// than sending as a Cypher parameter, so it is held to Neo4j's own grammar
/// for database names: 3 to 63 characters, an ASCII letter first, then ASCII
/// letters, digits, dots and dashes, and not ending in a dot or a dash. Every
/// character that could change the meaning of the path (`/`, `?`, `#`, `%`,
/// whitespace, anything outside ASCII) is refused by that grammar.
pub fn check_database_name(name: &str) -> Result<(), String> {
    let len = name.chars().count();
    if !(3..=63).contains(&len) {
        return Err(format!(
            "a Neo4j database name is 3 to 63 characters long; this one is {len}"
        ));
    }
    let first = name.chars().next().unwrap_or('\0');
    if !first.is_ascii_alphabetic() {
        return Err("a Neo4j database name starts with an ASCII letter".into());
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '.' || *c == '-'))
    {
        return Err(format!(
            "a Neo4j database name holds only ASCII letters, digits, dots and dashes; found {bad:?}"
        ));
    }
    if name.ends_with('.') || name.ends_with('-') {
        return Err("a Neo4j database name does not end in a dot or a dash".into());
    }
    Ok(())
}

/// One Cypher statement and its parameters. Agent-controlled message content
/// stays in parameters, never interpolated into executable Cypher.
struct Statement {
    cypher: String,
    parameters: serde_json::Map<String, serde_json::Value>,
}

fn statement(cypher: &str, parameters: &[(&str, &str)]) -> Statement {
    Statement {
        cypher: cypher.to_string(),
        parameters: parameters
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_string(),
                    serde_json::Value::String((*v).to_string()),
                )
            })
            .collect(),
    }
}

/// Start background sync from GraphStore to Neo4j.
///
/// Spawns a tokio task that:
/// 1. Creates indices/constraints on first run
/// 2. Optionally does a full-state export
/// 3. Subscribes to broadcasts and writes changes
///
/// Returns a JoinHandle for the sync task.
pub fn start(store: Arc<GraphStore>, config: Neo4jSyncConfig) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("Starting Neo4j background sync to {}", config.uri);

        let mut rx = store.subscribe();

        loop {
            match rx.recv().await {
                Ok(update) => {
                    if let Err(e) = write_update_to_neo4j(&config, &update).await {
                        warn!("Neo4j sync error: {}", e);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        "Neo4j sync lagged by {} updates, \
                         doing full export",
                        n
                    );
                    if let Err(e) = full_export(&store, &config).await {
                        error!("Neo4j full export failed: {}", e);
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    info!(
                        "GraphStore broadcast closed, \
                           stopping Neo4j sync"
                    );
                    break;
                }
            }
        }
    })
}

/// Build the statement that mirrors one graph update, or `None` for an update
/// the mirror does not carry.
fn statement_for_update(update: &crate::GraphUpdate) -> Option<Statement> {
    use crate::GraphUpdate as GU;

    match update {
        GU::NodeCreated { id, event, scope } => {
            let content = event.text.clone().unwrap_or_default();
            let agent = event.agent.clone().unwrap_or_default();
            // Canonical proto-Role -> lowercase string via the shared
            // helper (explicit), not Debug-derive.
            let role = event
                .role
                .and_then(sasy_common::MessageRole::from_proto_opt)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            debug!("Neo4j sync: node_created {}", id);
            Some(statement(
                "MERGE (m:Message {id: $id, tenant: $tenant}) \
                 SET m.content = $content, m.role = $role, \
                 m.agent = $agent, \
                 m.synced_at = datetime()",
                &[
                    ("id", id),
                    ("tenant", scope.tenant()),
                    ("content", &content),
                    ("role", &role),
                    ("agent", &agent),
                ],
            ))
        }
        GU::EdgeCreated {
            source,
            destination,
            scope,
            ..
        } => {
            debug!("Neo4j sync: edge_created {} → {}", source, destination);
            Some(statement(
                "MATCH (s:Message {id: $source, tenant: $tenant}) \
                 MATCH (d:Message {id: $destination, tenant: $tenant}) \
                 MERGE (d)-[:DEPENDS_ON]->(s)",
                &[
                    ("source", source),
                    ("destination", destination),
                    ("tenant", scope.tenant()),
                ],
            ))
        }
        // Deletions are not mirrored. `NodeDeleted` / `EdgeDeleted` carry no
        // `SessionScope`, so there is no tenant to key the match on, and a
        // match on the id alone would delete every tenant's node with that id
        // — the exact cross-tenant write the `(id, tenant)` key exists to
        // prevent. Nothing is lost today: the graph store never broadcasts
        // either variant, and it announces session teardown as `DropSession`,
        // which the mirror has always skipped.
        _ => None,
    }
}

/// The statement that mirrors one node from a FULL export.
///
/// Named, rather than built inline in [`full_export`], so the `(id, tenant)`
/// MERGE key and the parameter list are pinned by a test. The full export runs
/// on every sync restart, so a tenant dropped here would re-collide every
/// tenant's nodes on the next restart, whatever the incremental path does.
fn export_node_statement(tenant: &str, id: &str, content: &str, agent: &str) -> Statement {
    statement(
        "MERGE (m:Message {id: $id, tenant: $tenant}) \
         SET m.content = $content, m.agent = $agent",
        &[
            ("id", id),
            ("tenant", tenant),
            ("content", content),
            ("agent", agent),
        ],
    )
}

/// The statement that mirrors one edge from a FULL export. Same reason to be
/// named as [`export_node_statement`].
fn export_edge_statement(tenant: &str, source: &str, destination: &str) -> Statement {
    statement(
        "MATCH (s:Message {id: $source, tenant: $tenant}) \
         MATCH (d:Message {id: $destination, tenant: $tenant}) \
         MERGE (d)-[:DEPENDS_ON]->(s)",
        &[
            ("source", source),
            ("destination", destination),
            ("tenant", tenant),
        ],
    )
}

/// Write a single graph update to Neo4j.
async fn write_update_to_neo4j(
    config: &Neo4jSyncConfig,
    update: &crate::GraphUpdate,
) -> Result<(), anyhow::Error> {
    match statement_for_update(update) {
        Some(s) => execute_cypher(config, &s).await,
        None => Ok(()),
    }
}

/// Full state export: write all nodes and edges to Neo4j.
async fn full_export(store: &GraphStore, config: &Neo4jSyncConfig) -> Result<(), anyhow::Error> {
    let (per_scope, _seq) = store.get_full_state_by_scope()?;

    let (nodes, edges): (usize, usize) = per_scope
        .iter()
        .fold((0, 0), |(n, e), (_, ev, ed)| (n + ev.len(), e + ed.len()));
    info!("Neo4j full export: {} nodes, {} edges", nodes, edges);

    // Grouped by scope rather than flattened, because the tenant is a
    // property of the node being written and only the scope knows it.
    for (scope, events, edges) in &per_scope {
        let tenant = scope.tenant();

        for event in events {
            let id = event.id.clone().unwrap_or_default();
            let content = event.text.clone().unwrap_or_default();
            let agent = event.agent.clone().unwrap_or_default();

            let s = export_node_statement(tenant, &id, &content, &agent);
            execute_cypher(config, &s).await?;
        }

        for edge in edges {
            let s = export_edge_statement(tenant, &edge.source, &edge.destination);
            execute_cypher(config, &s).await?;
        }
    }

    info!("Neo4j full export complete");
    Ok(())
}

/// The HTTP transactional-endpoint body for one statement.
fn request_body(statement: &Statement) -> serde_json::Value {
    serde_json::json!({
        "statements": [{
            "statement": statement.cypher,
            "parameters": statement.parameters,
        }]
    })
}

/// Execute a Cypher statement against Neo4j.
///
/// Uses Neo4j's HTTP transaction endpoint.
async fn execute_cypher(
    config: &Neo4jSyncConfig,
    statement: &Statement,
) -> Result<(), anyhow::Error> {
    // Use Neo4j HTTP transactional API
    let url = config
        .uri
        .replace("bolt://", "http://")
        .replace("neo4j://", "http://")
        .replace(":7687", ":7474");
    let endpoint = format!("{}/db/{}/tx/commit", url, config.database);

    let client = reqwest::Client::new();
    let resp = client
        .post(&endpoint)
        .basic_auth(&config.username, Some(&config.password))
        .json(&request_body(statement))
        .send()
        .await?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Neo4j HTTP error: {}", text));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::observability::Event;
    use sasy_common::SessionScope;

    fn node_created(tenant: &str, id: &str, text: &str) -> crate::GraphUpdate {
        crate::GraphUpdate::NodeCreated {
            id: id.to_string(),
            event: Event {
                text: Some(text.to_string()),
                agent: Some("agent-1".into()),
                role: None,
                id: Some(id.to_string()),
                tools: vec![],
                derived_from: None,
                principal: None,
                entity: None,
                metadata: None,
            },
            scope: SessionScope::new(tenant, "conv"),
        }
    }

    /// Message content reaches Neo4j as a parameter, never as text spliced
    /// into the statement. The quote, the backslash and the `$` are the three
    /// characters a hand-escaped statement gets wrong: `'` would be escaped by
    /// hand, `\` would escape that escape, and `$` would name a parameter if it
    /// landed in the statement.
    #[test]
    fn message_content_travels_as_a_parameter_and_never_in_the_statement() {
        let hostile = r"it's a \ and a $tenant";
        let s = statement_for_update(&node_created("acme", "n1", hostile))
            .expect("a created node is mirrored");

        assert_eq!(
            s.parameters.get("content").and_then(|v| v.as_str()),
            Some(hostile),
            "the content must arrive verbatim, unescaped"
        );
        assert!(
            !s.cypher.contains(hostile) && !s.cypher.contains(r"it's"),
            "no part of the content may appear in the statement: {}",
            s.cypher
        );

        let body = request_body(&s);
        let sent = &body["statements"][0];
        assert_eq!(sent["statement"].as_str(), Some(s.cypher.as_str()));
        assert_eq!(
            sent["parameters"]["content"].as_str(),
            Some(hostile),
            "the request body must carry the value in `parameters`"
        );
    }

    /// Two tenants may use the same message id. The mirror is one database
    /// for the whole binary, so the MERGE key carries the tenant — otherwise
    /// the second tenant's write lands on the first tenant's node and
    /// overwrites its content.
    #[test]
    fn the_same_node_id_in_two_tenants_merges_as_two_nodes() {
        let acme = statement_for_update(&node_created("acme", "n1", "acme text")).unwrap();
        let other = statement_for_update(&node_created("other", "n1", "other text")).unwrap();

        assert!(
            acme.cypher
                .contains("MERGE (m:Message {id: $id, tenant: $tenant})"),
            "the MERGE key must name both: {}",
            acme.cypher
        );
        assert_eq!(
            acme.cypher, other.cypher,
            "one statement, two parameter sets"
        );
        assert_eq!(
            acme.parameters.get("id"),
            other.parameters.get("id"),
            "same id …"
        );
        assert_ne!(
            acme.parameters.get("tenant"),
            other.parameters.get("tenant"),
            "… different tenant, so the two writes cannot collide"
        );
    }

    /// Edges are matched on both ends' `(id, tenant)`, so a dependency never
    /// attaches to a same-named node belonging to somebody else.
    #[test]
    fn an_edge_matches_both_endpoints_within_one_tenant() {
        let update = crate::GraphUpdate::EdgeCreated {
            source: "n1".into(),
            destination: "n2".into(),
            kind: crate::types::EdgeKindSer::DependsOn,
            message_index: None,
            proximal: None,
            scope: SessionScope::new("acme", "conv"),
            principal: None,
            entity: None,
        };
        let s = statement_for_update(&update).expect("a created edge is mirrored");
        assert!(s
            .cypher
            .contains("(s:Message {id: $source, tenant: $tenant})"));
        assert!(s
            .cypher
            .contains("(d:Message {id: $destination, tenant: $tenant})"));
        assert_eq!(
            s.parameters.get("tenant").and_then(|v| v.as_str()),
            Some("acme")
        );
    }

    /// The full export — the path that runs on every sync restart — keys its
    /// nodes the same way the incremental path does, and carries content as a
    /// parameter. Two tenants using one message id stay two nodes.
    #[test]
    fn a_full_export_node_carries_its_tenant_in_the_merge_key() {
        let hostile = r"it's a \ and a $tenant";
        let acme = export_node_statement("acme", "n1", hostile, "agent-1");
        let other = export_node_statement("other", "n1", "other text", "agent-1");

        assert!(
            acme.cypher
                .contains("MERGE (m:Message {id: $id, tenant: $tenant})"),
            "the MERGE key must name both: {}",
            acme.cypher
        );
        assert!(
            !acme.cypher.contains(hostile),
            "content must never reach the statement: {}",
            acme.cypher
        );
        assert_eq!(
            acme.parameters.get("content").and_then(|v| v.as_str()),
            Some(hostile),
            "content travels verbatim as a parameter"
        );
        assert_eq!(
            acme.cypher, other.cypher,
            "one statement, two parameter sets"
        );
        assert_ne!(
            acme.parameters.get("tenant"),
            other.parameters.get("tenant"),
            "different tenant, so the two writes cannot collide"
        );
    }

    /// Exported edges match both endpoints within one tenant, so a restart's
    /// re-export cannot attach a dependency to another tenant's node.
    #[test]
    fn a_full_export_edge_matches_both_endpoints_within_one_tenant() {
        let s = export_edge_statement("acme", "n1", "n2");
        assert!(s
            .cypher
            .contains("(s:Message {id: $source, tenant: $tenant})"));
        assert!(s
            .cypher
            .contains("(d:Message {id: $destination, tenant: $tenant})"));
        assert_eq!(
            s.parameters.get("tenant").and_then(|v| v.as_str()),
            Some("acme")
        );
    }

    /// The export reads the store grouped by scope, because that is the only
    /// place the tenant survives. Two tenants come back as two groups with
    /// their own rows — flattened, there would be nothing left to stamp.
    #[test]
    fn the_store_hands_the_export_each_scope_with_its_own_rows() {
        use sasy_common::observability::Event;

        let store = crate::GraphStore::new(None).expect("in-memory store");
        let event = |id: &str, text: &str| Event {
            text: Some(text.to_string()),
            agent: Some("agent-1".into()),
            role: None,
            id: Some(id.to_string()),
            tools: vec![],
            derived_from: None,
            principal: None,
            entity: None,
            metadata: None,
        };
        let acme = SessionScope::new("acme", "conv");
        let other = SessionScope::new("other", "conv");
        store
            .merge_events(&acme, None, vec![event("n1", "acme text")])
            .unwrap();
        store
            .merge_events(&other, None, vec![event("n1", "other text")])
            .unwrap();

        let (per_scope, _seq) = store.get_full_state_by_scope().unwrap();
        let rows = |scope: &SessionScope| {
            per_scope
                .iter()
                .find(|(s, _, _)| s == scope)
                .map(|(_, events, _)| events.len())
        };
        assert_eq!(rows(&acme), Some(1), "acme's row belongs to acme");
        assert_eq!(rows(&other), Some(1), "other's row belongs to other");
    }

    #[test]
    fn config_default_fields() {
        let config = Neo4jSyncConfig {
            uri: "bolt://localhost:7687".into(),
            username: "neo4j".into(),
            password: "test".into(),
            database: "neo4j".into(),
        };
        assert_eq!(config.uri, "bolt://localhost:7687");
        assert_eq!(config.database, "neo4j");
    }

    #[test]
    fn a_database_name_is_accepted_only_in_the_shape_neo4j_defines() {
        for ok in ["neo4j", "system", "my-db.v2", "abc", "A1-b2.c3"] {
            assert_eq!(check_database_name(ok), Ok(()), "{ok:?} is a valid name");
        }
        let too_long = "a".repeat(64);
        for bad in [
            "",
            "ab",
            "1abc",
            "-abc",
            "abc.",
            "abc-",
            "neo4j/../system",
            "neo4j?x=1",
            "neo4j#frag",
            "db name",
            "neo%2Fj",
            "caf\u{e9}db",
            too_long.as_str(),
        ] {
            assert!(check_database_name(bad).is_err(), "{bad:?} must be refused");
        }
    }
}

//! Immutable full snapshots. Resolution and validation happen under one shard
//! lock; only a completely validated batch enters the existing durable writer.
use super::*;
use crate::types::MessageNode;
use prost::Message;
use sasy_common::observability::EventSnapshot;
use sha2::{Digest, Sha256};

const PREFIX: &str = "sasy:mv1:";

pub(super) fn is_immutable(id: &str) -> bool {
    id.starts_with(PREFIX)
}

fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidSnapshot(message.into())
}

fn fingerprint<T: serde::Serialize>(value: &T) -> Result<String, GraphError> {
    let digest = Sha256::digest(serde_json::to_vec(value)?);
    Ok(format!("{digest:x}"))
}

/// Canonical cross-language digest: protobuf preserves optional presence and
/// repeated tool order; neither origin aliases nor authenticated principals are
/// content. Prost decoding discards unknown fields before this projection.
fn event_content_hash(mut event: Event) -> [u8; 32] {
    event.id = None;
    event.principal = None;
    let mut digest = Sha256::new();
    digest.update(b"sasy:event-content:v1\0");
    digest.update(event.encode_to_vec());
    digest.finalize().into()
}

fn stored_content_hash(s: &mut SessionShard, id: &str) -> Result<[u8; 32], GraphError> {
    if let Some(hash) = s.content_hashes.get(id) {
        return Ok(*hash);
    }
    let event = convert::message_to_event(
        message(s, id).ok_or_else(|| invalid("base version is not present in this scope"))?,
    );
    let hash = event_content_hash(event);
    s.content_hashes.insert(id.to_owned(), hash);
    Ok(hash)
}

fn message<'a>(s: &'a SessionShard, id: &str) -> Option<&'a MessageNode> {
    s.msg_idx
        .get(id)
        .and_then(|i| match s.graph.node_weight(*i) {
            Some(NodeData::Message(message)) => Some(message),
            _ => None,
        })
}

fn edge_data(scope: &SessionScope, principal: Option<&str>, edge: &Edge) -> EdgeData {
    EdgeData {
        kind: EdgeKindSer::DependsOn,
        message_index: edge.message_index,
        proximal: edge.proximal,
        scope: scope.clone(),
        principal: principal.map(str::to_owned),
        entity: edge.entity.clone(),
    }
}

fn incoming(s: &SessionShard, id: &str) -> Vec<(String, EdgeData)> {
    let Some(index) = s.msg_idx.get(id) else {
        return Vec::new();
    };
    // Graph storage reverses message dependencies: destination -> source.
    let mut edges: Vec<_> = s
        .graph
        .edges(*index)
        .filter(|edge| edge.weight().kind == EdgeKindSer::DependsOn)
        .map(|edge| (node_id(&s.graph, edge.target()), edge.weight().clone()))
        .collect();
    edges.sort_by(|a, b| a.0.cmp(&b.0));
    edges
}

pub(super) fn validate_legacy_events(
    s: &SessionShard,
    scope: &SessionScope,
    principal: Option<&str>,
    events: &[Event],
) -> Result<(), GraphError> {
    for event in events {
        let Some(id) = event.id.as_deref().filter(|id| is_immutable(id)) else {
            continue;
        };
        let desired = convert::event_to_message(event, id, scope.clone(), principal);
        if message(s, id) != Some(&desired) {
            return Err(GraphError::ImmutableViolation(
                "version IDs require ResolveEvents; legacy content updates are forbidden".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_legacy_edges(
    s: &SessionShard,
    scope: &SessionScope,
    principal: Option<&str>,
    edges: &[Edge],
) -> Result<(), GraphError> {
    for edge in edges.iter().filter(|e| is_immutable(&e.destination)) {
        let key = EdgeKey {
            source: edge.source.clone(),
            destination: edge.destination.clone(),
            kind: EdgeKindSer::DependsOn,
        };
        let existing = s.edge_idx.get(&key).and_then(|i| s.graph.edge_weight(*i));
        if existing != Some(&edge_data(scope, principal, edge)) {
            return Err(GraphError::ImmutableViolation(
                "incoming version dependencies cannot be changed".into(),
            ));
        }
    }
    Ok(())
}

struct Resolved {
    id: String,
    event: Event,
    dependencies: Vec<(String, EdgeData)>,
    exists: bool,
}

impl GraphStore {
    /// Resolve full replacement snapshots to immutable IDs in input order.
    /// Unchanged replays produce no storage writes, sequence bumps or broadcasts.
    pub fn resolve_events(
        &self,
        scope: &SessionScope,
        principal: Option<&str>,
        snapshots: Vec<EventSnapshot>,
    ) -> Result<Vec<String>, GraphError> {
        if snapshots.is_empty() {
            return Ok(Vec::new());
        }
        let shard = self.get_or_create_shard(scope);
        let mut s = shard.write();
        if s.diverged {
            return Err(diverged_error(scope));
        }
        let mut aliases = HashMap::with_capacity(snapshots.len());
        for (index, snapshot) in snapshots.iter().enumerate() {
            let event = snapshot
                .event
                .as_ref()
                .ok_or_else(|| invalid("event is required"))?;
            let alias = event
                .id
                .as_deref()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| invalid("nonempty event.id origin is required"))?;
            if is_immutable(alias) || aliases.insert(alias.to_owned(), index).is_some() {
                return Err(invalid(
                    "origin aliases must be unique and outside the immutable namespace",
                ));
            }
            if event
                .role
                .is_some_and(|role| sasy_common::MessageRole::from_proto_opt(role).is_none())
            {
                return Err(invalid("unknown event role"));
            }
            if let Some(hash) = &snapshot.content_hash {
                let reference = Event {
                    id: event.id.clone(),
                    ..Default::default()
                };
                if hash.len() != 32
                    || event != &reference
                    || snapshot.base_id.is_none()
                    || !snapshot.reuse_dependencies
                    || !snapshot.dependencies.is_empty()
                {
                    return Err(invalid("compact references require a 32-byte content hash, event origin only, base_id and reused dependencies"));
                }
            }
            if snapshot.reuse_dependencies
                && (snapshot.base_id.is_none() || !snapshot.dependencies.is_empty())
            {
                return Err(invalid(
                    "reuse_dependencies requires a base and no explicit dependencies",
                ));
            }
        }
        // Kahn's algorithm avoids recursion and handles large transcript batches.
        let mut indegree = vec![0usize; snapshots.len()];
        let mut children = vec![Vec::new(); snapshots.len()];
        for (index, snapshot) in snapshots.iter().enumerate() {
            let alias = snapshot.event.as_ref().unwrap().id.as_deref().unwrap();
            let mut sources = HashSet::new();
            for edge in &snapshot.dependencies {
                if edge.destination != alias || !sources.insert(edge.source.as_str()) {
                    return Err(invalid(
                        "dependencies must target their origin and have unique sources",
                    ));
                }
                if let Some(&parent) = aliases.get(&edge.source) {
                    indegree[index] += 1;
                    children[parent].push(index);
                } else if !is_immutable(&edge.source) || message(&s, &edge.source).is_none() {
                    return Err(invalid("dependency source must be a batch alias or immutable version in this scope"));
                }
            }
        }
        let mut ready: VecDeque<_> = indegree
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n == 0).then_some(i))
            .collect();
        let mut resolved: Vec<Option<Resolved>> = (0..snapshots.len()).map(|_| None).collect();
        let mut order = Vec::with_capacity(snapshots.len());
        while let Some(index) = ready.pop_front() {
            let snapshot = &snapshots[index];
            let event = snapshot.event.as_ref().unwrap();
            let alias = event.id.as_deref().unwrap();
            let origin = fingerprint(&("sasy:event-origin:v1", scope, principal, alias))?;
            let origin_prefix = format!("{PREFIX}{origin}:");
            let desired = convert::event_to_message(event, "", scope.clone(), principal);
            let base = match snapshot.base_id.as_deref() {
                Some(id) => {
                    if !id.starts_with(&origin_prefix) || id.len() != origin_prefix.len() + 64 {
                        return Err(invalid(
                            "base must be an immutable version of this origin, scope and principal",
                        ));
                    }
                    Some(
                        message(&s, id)
                            .ok_or_else(|| invalid("base version is not present in this scope"))?,
                    )
                }
                None => None,
            };
            if let Some(expected) = &snapshot.content_hash {
                let id = snapshot.base_id.as_deref().unwrap();
                if stored_content_hash(&mut s, id)?.as_slice() != expected.as_slice() {
                    return Err(invalid(
                        "compact reference content hash does not match its immutable version",
                    ));
                }
                // The base's identity, scope and existence were checked above.
                // It cannot change under this lock. No body clone, dependency
                // traversal or durable write is needed on the cached path.
                resolved[index] = Some(Resolved {
                    id: id.to_owned(),
                    event: Event::default(),
                    dependencies: Vec::new(),
                    exists: true,
                });
            } else {
                let mut dependencies = Vec::new();
                if snapshot.reuse_dependencies {
                    let base = base.unwrap();
                    let mut content = base.clone();
                    content.id.clear();
                    if content != desired {
                        return Err(invalid(
                            "reuse_dependencies does not explain changed event content",
                        ));
                    }
                    dependencies = incoming(&s, &base.id);
                } else {
                    for edge in &snapshot.dependencies {
                        let source = match aliases.get(&edge.source) {
                            Some(&parent) => resolved[parent].as_ref().unwrap().id.clone(),
                            None => edge.source.clone(),
                        };
                        dependencies.push((source, edge_data(scope, principal, edge)));
                    }
                    dependencies.sort_by(|a, b| a.0.cmp(&b.0));
                    if dependencies.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                        return Err(invalid(
                            "multiple references resolved to one dependency source",
                        ));
                    }
                }
                let unchanged_base = base.filter(|base| {
                    let mut previous = (*base).clone();
                    previous.id.clear();
                    previous == desired && incoming(&s, &base.id) == dependencies
                });
                let id = if let Some(base) = unchanged_base {
                    base.id.clone()
                } else {
                    format!(
                        "{origin_prefix}{}",
                        fingerprint(&(
                            "sasy:event-version:v1",
                            snapshot.base_id.as_deref(),
                            &desired,
                            &dependencies
                        ))?
                    )
                };
                let mut actual = desired;
                actual.id = id.clone();
                let exists = if let Some(existing) = message(&s, &id) {
                    if existing != &actual || incoming(&s, &id) != dependencies {
                        return Err(GraphError::ContentHashAmbiguity(
                            "immutable version key contains a different snapshot".into(),
                        ));
                    }
                    true
                } else {
                    false
                };
                resolved[index] = Some(Resolved {
                    id,
                    event: convert::message_to_event(&actual),
                    dependencies,
                    exists,
                });
            }
            order.push(index);
            for &child in &children[index] {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    ready.push_back(child);
                }
            }
        }
        if order.len() != snapshots.len() {
            return Err(invalid("cyclic snapshot dependencies"));
        }
        // Nothing above mutated the shard. Every endpoint, base, hash and batch
        // relationship is now checked before entering the durable write path.
        let ids = resolved
            .iter()
            .map(|r| r.as_ref().unwrap().id.clone())
            .collect();
        let mut batch = self.rocks.new_batch();
        let mut updates = Vec::new();
        let mut new_nodes = Vec::new();
        let mut replay = Vec::new();
        let mut edges = Vec::new();
        for index in order {
            let item = resolved[index].as_ref().unwrap();
            if item.exists {
                continue;
            }
            if let Err(error) = self.upsert_event_into_shard(
                &mut s,
                &item.event,
                scope,
                principal,
                &mut batch,
                &mut updates,
                &mut new_nodes,
                &mut replay,
                true,
            ) {
                if !updates.is_empty() {
                    s.diverged = true;
                }
                return Err(error);
            }
            edges.extend(item.dependencies.iter().map(|(source, data)| Edge {
                source: source.clone(),
                destination: item.id.clone(),
                message_index: data.message_index,
                proximal: data.proximal,
                principal: data.principal.clone(),
                entity: data.entity.clone(),
            }));
        }
        if let Err(error) = self.merge_dependencies_into_shard(
            &mut s,
            &edges,
            principal,
            &mut batch,
            &mut updates,
            &mut replay,
            scope,
        ) {
            s.diverged = true;
            return Err(error);
        }
        let seq = self.global_sequence.load(Ordering::SeqCst);
        self.finalize_shard_write(s, batch, &new_nodes, scope, updates, replay, seq)?;
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::observability::{Role, Tool};

    fn snapshot(alias: &str, text: &str, sources: &[&str]) -> EventSnapshot {
        EventSnapshot {
            event: Some(Event {
                id: Some(alias.into()),
                text: Some(text.into()),
                role: Some(Role::User as i32),
                ..Default::default()
            }),
            base_id: None,
            reuse_dependencies: false,
            content_hash: None,
            dependencies: sources
                .iter()
                .map(|source| Edge {
                    source: (*source).into(),
                    destination: alias.into(),
                    ..Default::default()
                })
                .collect(),
        }
    }
    fn scope() -> SessionScope {
        SessionScope::new("tenant", "session")
    }
    fn resolve(store: &GraphStore, snapshots: Vec<EventSnapshot>) -> Vec<String> {
        store
            .resolve_events(&scope(), Some("writer"), snapshots)
            .unwrap()
    }

    fn reference(original: &EventSnapshot, id: &str) -> EventSnapshot {
        EventSnapshot {
            event: Some(Event {
                id: original.event.as_ref().unwrap().id.clone(),
                ..Default::default()
            }),
            base_id: Some(id.into()),
            reuse_dependencies: true,
            content_hash: Some(event_content_hash(original.event.clone().unwrap()).to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn compact_references_are_quiet_cached_and_support_mixed_aliases() {
        let store = GraphStore::new(None).unwrap();
        let original = snapshot("input", "question", &[]);
        let id = resolve(&store, vec![original.clone()])[0].clone();
        let compact = reference(&original, &id);
        assert!(store
            .shard_ro(&scope())
            .unwrap()
            .read()
            .content_hashes
            .is_empty());
        let mixed = resolve(
            &store,
            vec![snapshot("output", "answer", &["input"]), compact.clone()],
        );
        assert_eq!(mixed[1], id);
        let shard = store.shard_ro(&scope()).unwrap();
        assert_eq!(incoming(&shard.read(), &mixed[0])[0].0, id);
        assert_eq!(shard.read().content_hashes.len(), 1);
        let seq = store.session_sequence(&scope());
        let mut rx = store.subscribe();
        store.rocks.set_fail_writes_for_test(true);
        assert_eq!(resolve(&store, vec![compact.clone()]), vec![id.clone()]);
        assert_eq!(resolve(&store, vec![compact]), vec![id]);
        assert_eq!(store.session_sequence(&scope()), seq);
        assert!(rx.try_recv().is_err());
    }

    /// `metadata` is part of the event the content hash covers, so a node
    /// that lost it on the way through the store would no longer answer to
    /// the mark its writer holds.
    #[test]
    fn metadata_survives_the_store_and_is_covered_by_the_content_hash() {
        let store = GraphStore::new(None).unwrap();
        let mut original = snapshot("input", "question", &[]);
        original.event.as_mut().unwrap().metadata = Some("{\"parts\":[1]}".into());
        let id = resolve(&store, vec![original.clone()])[0].clone();
        assert_eq!(
            resolve(&store, vec![reference(&original, &id)]),
            vec![id.clone()]
        );

        let mut altered = original.clone();
        altered.event.as_mut().unwrap().metadata = Some("{\"parts\":[2]}".into());
        assert!(matches!(
            store.resolve_events(&scope(), Some("writer"), vec![reference(&altered, &id)]),
            Err(GraphError::InvalidSnapshot(_))
        ));
    }

    /// An event that sets no `metadata` hashes exactly as it did before the
    /// field existed, so graphs recorded without it stay referenceable. The
    /// digest is over the encoding of a fixed event, computed independently
    /// of this code.
    #[test]
    fn an_event_without_metadata_keeps_its_digest() {
        let event = Event {
            text: Some("question".into()),
            agent: Some("a".into()),
            role: Some(Role::User as i32),
            id: Some("ignored-origin".into()),
            entity: Some("e".into()),
            principal: Some("ignored-principal".into()),
            ..Default::default()
        };
        let digest: String = event_content_hash(event.clone())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            digest,
            "6c43d3d944c5e57d51fbd5ef43ca8c446926b01d57bd593a803d56788401f5ba"
        );
        // Unset is not the empty string: presence itself is encoded.
        let mut present = event.clone();
        present.metadata = Some(String::new());
        assert_ne!(event_content_hash(present), event_content_hash(event));
    }

    /// A snapshot that carries no metadata resolves to the immutable version
    /// id it resolved to before the field existed, so a writer replaying a
    /// pre-upgrade snapshot gets its own node back instead of a new one.
    ///
    /// The version id is a digest of the JSON of the resolved node; the
    /// expected digest here is taken over that JSON as it was written before
    /// `metadata` was added — spelled out, not produced by the types under
    /// test. A serialized `"metadata":null` would change it.
    #[test]
    fn a_metadata_free_snapshot_keeps_its_version_id() {
        let store = GraphStore::new(None).unwrap();
        let id = resolve(&store, vec![snapshot("input", "question", &[])])[0].clone();

        let before_the_field = concat!(
            r#"["sasy:event-version:v1",null,"#,
            r#"{"id":"","content":"question","role":"user","agent":null,"#,
            r#""tools_json":null,"derived_from_json":null,"#,
            r#""scope":{"tenant":"tenant","session":"session"},"#,
            r#""principal":"writer","entity":null},[]]"#
        );
        let expected: String = Sha256::digest(before_the_field.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert!(
            id.ends_with(&expected),
            "a metadata-free snapshot must keep its pre-field version id; got {id}, \
             expected it to end with {expected}"
        );
    }

    #[test]
    fn invalid_compact_references_fail_before_any_graph_write() {
        let store = GraphStore::new(None).unwrap();
        let original = snapshot("origin", "value", &[]);
        let id = resolve(&store, vec![original.clone()])[0].clone();
        let valid = reference(&original, &id);
        let mut invalids = Vec::new();
        for hash in [vec![], vec![1; 31], vec![1; 32], vec![1; 33]] {
            let mut bad = valid.clone();
            bad.content_hash = Some(hash);
            invalids.push(bad);
        }
        for event in [
            Event {
                id: Some("origin".into()),
                text: Some(String::new()),
                ..Default::default()
            },
            Event {
                id: Some("origin".into()),
                principal: Some("forged".into()),
                ..Default::default()
            },
            Event {
                id: Some("wrong-origin".into()),
                ..Default::default()
            },
        ] {
            let mut bad = valid.clone();
            bad.event = Some(event);
            invalids.push(bad);
        }
        let mut bad = valid.clone();
        bad.base_id = None;
        invalids.push(bad);
        let mut bad = valid.clone();
        bad.base_id.as_mut().unwrap().push('x');
        invalids.push(bad);
        let mut bad = valid.clone();
        bad.reuse_dependencies = false;
        invalids.push(bad);
        let mut bad = valid.clone();
        bad.dependencies.push(Edge {
            source: id,
            destination: "origin".into(),
            ..Default::default()
        });
        invalids.push(bad);
        let seq = store.session_sequence(&scope());
        let mut rx = store.subscribe();
        for bad in invalids {
            assert!(matches!(
                store.resolve_events(
                    &scope(),
                    Some("writer"),
                    vec![snapshot("new", "must not land", &[]), bad]
                ),
                Err(GraphError::InvalidSnapshot(_))
            ));
            assert_eq!(store.session_counts(&scope()), (1, 0));
            assert_eq!(store.session_sequence(&scope()), seq);
            assert!(rx.try_recv().is_err());
        }
        for other in [
            SessionScope::new("other", "session"),
            SessionScope::new("tenant", "other"),
        ] {
            assert!(store
                .resolve_events(&other, Some("writer"), vec![valid.clone()])
                .is_err());
            assert_eq!(store.session_counts(&other), (0, 0));
        }
        assert!(store
            .resolve_events(&scope(), Some("different"), vec![valid])
            .is_err());
    }

    #[test]
    fn compact_cache_is_rebuilt_after_restart_and_discarded_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let original = snapshot("origin", "durable", &[]);
        let compact;
        {
            let store = GraphStore::new(Some(path)).unwrap();
            let id = resolve(&store, vec![original.clone()])[0].clone();
            compact = reference(&original, &id);
            assert_eq!(resolve(&store, vec![compact.clone()]), vec![id]);
        }
        let store = GraphStore::new(Some(path)).unwrap();
        let shard = store.shard_ro(&scope()).unwrap();
        assert!(shard.read().content_hashes.is_empty());
        let seq = store.session_sequence(&scope());
        let mut rx = store.subscribe();
        store.rocks.set_fail_writes_for_test(true);
        assert_eq!(
            resolve(&store, vec![compact.clone()]),
            vec![compact.base_id.clone().unwrap()]
        );
        assert_eq!(store.session_sequence(&scope()), seq);
        assert!(rx.try_recv().is_err());
        store.rocks.set_fail_writes_for_test(false);
        store.drop_session(&scope()).unwrap();
        assert!(shard.read().content_hashes.is_empty());
        assert!(store
            .resolve_events(&scope(), Some("writer"), vec![compact.clone()])
            .is_err());
        let id = resolve(&store, vec![original])[0].clone();
        assert_eq!(resolve(&store, vec![compact]), vec![id.clone()]);
        shard.write().diverged = true;
        assert!(store
            .resolve_events(
                &scope(),
                Some("writer"),
                vec![reference(&snapshot("origin", "durable", &[]), &id)]
            )
            .is_err());
    }

    #[test]
    fn content_digest_matches_shared_cross_language_vectors() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|root| root.join("tests/fixtures/message-content-hashes.json"))
            .find(|p| p.is_file())
            .expect("shared content hash vectors");
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for vector in fixture["vectors"].as_array().unwrap() {
            let hex = vector["canonical_proto_hex"].as_str().unwrap();
            let wire: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            let mut event = Event::decode(wire.as_slice()).unwrap();
            assert_eq!(event.encode_to_vec(), wire);
            event.id = Some("ignored-origin".into());
            event.principal = Some("ignored-principal".into());
            let hash: String = event_content_hash(event.clone())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                hash,
                vector["sha256"].as_str().unwrap(),
                "{}",
                vector["name"]
            );
            // Unknown scalar fields, including nested tools, do not enter the hash.
            let mut unknown = event.encode_to_vec();
            unknown.extend_from_slice(&[0xa0, 0x06, 0x01]);
            assert_eq!(
                event_content_hash(Event::decode(unknown.as_slice()).unwrap()),
                event_content_hash(event)
            );
        }
        let nested_unknown = Event::decode(&[0x2a, 0x03, 0xa0, 0x06, 0x01][..]).unwrap();
        assert_eq!(
            event_content_hash(nested_unknown),
            event_content_hash(Event {
                tools: vec![Tool::default()],
                ..Default::default()
            })
        );
    }

    #[test]
    fn topology_remaps_aliases_and_retries_without_any_write_or_broadcast() {
        let store = GraphStore::new(None).unwrap();
        let request = vec![
            snapshot("output", "answer", &["input"]),
            snapshot("input", "question", &[]),
        ];
        let ids = resolve(&store, request.clone());
        let shard = store.shard_ro(&scope()).unwrap();
        assert_eq!(incoming(&shard.read(), &ids[0])[0].0, ids[1]);
        let sequence = store.session_sequence(&scope());
        let mut subscription = store.subscribe();
        store.rocks.set_fail_writes_for_test(true);
        assert_eq!(resolve(&store, request), ids);
        assert_eq!(store.session_sequence(&scope()), sequence);
        assert!(subscription.try_recv().is_err());
    }

    #[test]
    fn full_replacement_removes_fields_and_dependencies_without_inheriting_approval() {
        let store = GraphStore::new(None).unwrap();
        let mut approved = snapshot("output", "approved", &["input"]);
        let event = approved.event.as_mut().unwrap();
        event.agent = Some("reviewer".into());
        event.entity = Some("person".into());
        event.tools = vec![Tool {
            name: Some("approve".into()),
            arguments: Some("{}".into()),
        }];
        event.derived_from = Some(event.tools[0].clone());
        let old = resolve(&store, vec![snapshot("input", "request", &[]), approved]);
        let replacement = EventSnapshot {
            event: Some(Event {
                id: Some("output".into()),
                ..Default::default()
            }),
            base_id: Some(old[1].clone()),
            ..Default::default()
        };
        let next = resolve(&store, vec![replacement])[0].clone();
        let shard = store.shard_ro(&scope()).unwrap();
        let s = shard.read();
        let node = message(&s, &next).unwrap();
        assert!(node.content.is_none() && node.role.is_none() && node.agent.is_none());
        assert!(
            node.entity.is_none() && node.tools_json.is_none() && node.derived_from_json.is_none()
        );
        assert!(incoming(&s, &next).is_empty());
        assert_eq!(incoming(&s, &old[1]).len(), 1);
        assert!(message(&s, &old[1]).unwrap().derived_from_json.is_some());
    }

    #[test]
    fn reuse_requires_unchanged_content_and_explicit_unchanged_returns_base() {
        let store = GraphStore::new(None).unwrap();
        let old = resolve(
            &store,
            vec![
                snapshot("parent", "p", &[]),
                snapshot("child", "c", &["parent"]),
            ],
        );
        let mut item = snapshot("child", "c", &[]);
        item.base_id = Some(old[1].clone());
        item.reuse_dependencies = true;
        assert_eq!(resolve(&store, vec![item.clone()]), vec![old[1].clone()]);
        item.event.as_mut().unwrap().text = Some("changed".into());
        let sequence = store.session_sequence(&scope());
        assert!(matches!(
            store.resolve_events(&scope(), Some("writer"), vec![item]),
            Err(GraphError::InvalidSnapshot(_))
        ));
        assert_eq!(store.session_sequence(&scope()), sequence);
        let mut explicit = snapshot("child", "c", &[&old[0]]);
        explicit.base_id = Some(old[1].clone());
        assert_eq!(resolve(&store, vec![explicit]), vec![old[1].clone()]);
    }

    #[test]
    fn content_and_dependency_versions_preserve_aba_and_stale_base_branches() {
        let store = GraphStore::new(None).unwrap();
        let first = resolve(&store, vec![snapshot("origin", "a", &[])])[0].clone();
        let mut b = snapshot("origin", "b", &[]);
        b.base_id = Some(first.clone());
        let second = resolve(&store, vec![b.clone()])[0].clone();
        let mut a = snapshot("origin", "a", &[]);
        a.base_id = Some(second.clone());
        let third = resolve(&store, vec![a])[0].clone();
        assert_ne!(first, second);
        assert_ne!(first, third);
        assert_ne!(second, third);
        assert_eq!(resolve(&store, vec![b]), vec![second.clone()]);
        let parent = resolve(&store, vec![snapshot("parent", "p", &[])])[0].clone();
        let mut dep = snapshot("origin", "b", &[&parent]);
        dep.base_id = Some(second.clone());
        let fourth = resolve(&store, vec![dep])[0].clone();
        assert_ne!(fourth, second);
        let shard = store.shard_ro(&scope()).unwrap();
        assert_eq!(incoming(&shard.read(), &fourth)[0].0, parent);
    }

    #[test]
    fn dependency_order_is_not_identity_but_metadata_is() {
        let store = GraphStore::new(None).unwrap();
        let parents = resolve(
            &store,
            vec![snapshot("p", "p", &[]), snapshot("q", "q", &[])],
        );
        let original = snapshot("c", "c", &[&parents[0], &parents[1]]);
        let first = resolve(&store, vec![original.clone()])[0].clone();
        let mut reordered = original.clone();
        reordered.dependencies.reverse();
        assert_eq!(resolve(&store, vec![reordered]), vec![first.clone()]);
        let mut changed = original;
        changed.base_id = Some(first.clone());
        changed.dependencies[0].message_index = Some(2);
        changed.dependencies[0].entity = Some("new".into());
        assert_ne!(resolve(&store, vec![changed])[0], first);
    }

    #[test]
    fn origin_scope_and_authenticated_principal_separate_identity() {
        let store = GraphStore::new(None).unwrap();
        let event = snapshot("origin", "same", &[]);
        let first = resolve(&store, vec![event.clone()])[0].clone();
        let mut spoofed = event.clone();
        spoofed.event.as_mut().unwrap().principal = Some("spoof".into());
        assert_eq!(resolve(&store, vec![spoofed]), vec![first.clone()]);
        let mut other_origin = event.clone();
        other_origin.event.as_mut().unwrap().id = Some("another".into());
        let other = resolve(&store, vec![other_origin])[0].clone();
        assert_ne!(first, other);
        for scoped in [
            SessionScope::new("tenant", "other"),
            SessionScope::new("other", "session"),
        ] {
            assert_ne!(
                store
                    .resolve_events(&scoped, Some("writer"), vec![event.clone()])
                    .unwrap()[0],
                first
            );
        }
        assert_ne!(
            store
                .resolve_events(&scope(), Some("other"), vec![event.clone()])
                .unwrap()[0],
            first
        );
        let mut invalid_base = event;
        invalid_base.base_id = Some(other);
        assert!(store
            .resolve_events(&scope(), Some("writer"), vec![invalid_base])
            .is_err());
    }

    #[test]
    fn invalid_batch_edges_aliases_and_bases_are_atomic() {
        let cases = vec![
            vec![
                snapshot("good", "g", &[]),
                snapshot("bad", "b", &["missing"]),
            ],
            vec![snapshot("a", "a", &["b"]), snapshot("b", "b", &["a"])],
            vec![snapshot("a", "a", &[]), snapshot("a", "a", &[])],
            vec![snapshot("self", "s", &["self"])],
            vec![
                snapshot("a", "a", &[]),
                EventSnapshot {
                    base_id: Some("absent".into()),
                    ..snapshot("b", "b", &[])
                },
            ],
        ];
        for batch in cases {
            let store = GraphStore::new(None).unwrap();
            let mut rx = store.subscribe();
            assert!(store
                .resolve_events(&scope(), Some("writer"), batch)
                .is_err());
            assert_eq!(store.session_counts(&scope()), (0, 0));
            assert_eq!(store.session_sequence(&scope()), 0);
            assert!(rx.try_recv().is_err());
            assert!(!store.shard_ro(&scope()).unwrap().read().diverged);
        }
    }

    #[test]
    fn legacy_sources_are_not_frozen_by_reference() {
        let store = GraphStore::new(None).unwrap();
        store
            .merge_events(
                &scope(),
                Some("writer"),
                vec![snapshot("old", "mutable", &[]).event.unwrap()],
            )
            .unwrap();
        assert!(store
            .resolve_events(
                &scope(),
                Some("writer"),
                vec![snapshot("new", "n", &["old"])]
            )
            .is_err());
    }

    #[test]
    fn legacy_writes_cannot_mutate_or_create_version_namespace() {
        let store = GraphStore::new(None).unwrap();
        let ids = resolve(
            &store,
            vec![snapshot("p", "p", &[]), snapshot("c", "c", &["p"])],
        );
        let original = {
            let shard = store.shard_ro(&scope()).unwrap();
            let s = shard.read();
            convert::message_to_event(message(&s, &ids[1]).unwrap())
        };
        let sequence = store.session_sequence(&scope());
        assert_eq!(
            store
                .merge_events(&scope(), Some("writer"), vec![original.clone()])
                .unwrap(),
            vec![ids[1].clone()]
        );
        let mut changed = original;
        changed.text = Some("mutated".into());
        let good = snapshot("legacy", "should not land", &[]).event.unwrap();
        assert!(matches!(
            store.merge_events(
                &scope(),
                Some("writer"),
                vec![good.clone(), changed.clone()]
            ),
            Err(GraphError::ImmutableViolation(_))
        ));
        assert!(store
            .merge_events_with_dependencies(&scope(), Some("writer"), vec![good, changed], vec![])
            .is_err());
        let edge = Edge {
            source: ids[0].clone(),
            destination: ids[1].clone(),
            ..Default::default()
        };
        store
            .merge_dependencies(&scope(), Some("writer"), vec![edge.clone()])
            .unwrap();
        let mut changed_edge = edge;
        changed_edge.proximal = Some(true);
        assert!(store
            .merge_dependencies(&scope(), Some("writer"), vec![changed_edge])
            .is_err());
        assert!(store.delete_dependency(&scope(), &ids[0], &ids[1]).is_err());
        assert!(store
            .merge_events(
                &scope(),
                Some("writer"),
                vec![snapshot("sasy:mv1:forged", "forged", &[]).event.unwrap()]
            )
            .is_err());
        assert_eq!(store.session_sequence(&scope()), sequence);
        assert_eq!(store.session_counts(&scope()), (2, 1));
    }

    #[test]
    fn restart_preserves_content_addressing_and_noop_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let batch = vec![snapshot("p", "p", &[]), snapshot("c", "c", &["p"])];
        let ids = {
            let store = GraphStore::new(Some(path)).unwrap();
            resolve(&store, batch.clone())
        };
        let store = GraphStore::new(Some(path)).unwrap();
        let sequence = store.session_sequence(&scope());
        let mut rx = store.subscribe();
        assert_eq!(resolve(&store, batch), ids);
        assert_eq!(store.session_sequence(&scope()), sequence);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn failed_commit_quarantines_snapshot_reads_and_retries() {
        let store = GraphStore::new(None).unwrap();
        let mut rx = store.subscribe();
        store.rocks.set_fail_writes_for_test(true);
        let batch = vec![snapshot("p", "p", &[]), snapshot("c", "c", &["p"])];
        assert!(store
            .resolve_events(&scope(), Some("writer"), batch.clone())
            .is_err());
        store.rocks.set_fail_writes_for_test(false);
        assert!(store
            .resolve_events(&scope(), Some("writer"), batch)
            .is_err());
        assert!(store.get_full_state().is_err());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn large_unchanged_replay_is_silent() {
        let store = GraphStore::new(None).unwrap();
        let mut batch = Vec::new();
        for index in 0usize..4000 {
            let origin = format!("m{index}");
            let previous = format!("m{}", index.saturating_sub(1));
            let previous_ref = previous.as_str();
            batch.push(snapshot(
                &origin,
                "message",
                if index == 0 {
                    &[]
                } else {
                    std::slice::from_ref(&previous_ref)
                },
            ));
        }
        let ids = resolve(&store, batch.clone());
        let sequence = store.session_sequence(&scope());
        let mut rx = store.subscribe();
        store.rocks.set_fail_writes_for_test(true);
        assert_eq!(resolve(&store, batch), ids);
        assert_eq!(store.session_sequence(&scope()), sequence);
        assert_eq!(store.session_counts(&scope()), (4000, 3999));
        assert!(rx.try_recv().is_err());
    }
    #[test]
    fn stored_content_collision_is_refused_before_other_updates() {
        let store = GraphStore::new(None).unwrap();
        let request = snapshot("origin", "correct", &[]);
        let id = resolve(&store, vec![request.clone()])[0].clone();
        let shard = store.shard_ro(&scope()).unwrap();
        {
            let mut s = shard.write();
            let index = s.msg_idx[&id];
            if let Some(NodeData::Message(node)) = s.graph.node_weight_mut(index) {
                node.content = Some("corrupted".into());
            }
        }
        let sequence = store.session_sequence(&scope());
        assert!(matches!(
            store.resolve_events(
                &scope(),
                Some("writer"),
                vec![snapshot("new", "must not land", &[]), request]
            ),
            Err(GraphError::ContentHashAmbiguity(_))
        ));
        assert_eq!(store.session_sequence(&scope()), sequence);
        assert_eq!(store.session_counts(&scope()), (1, 0));
    }

    #[test]
    fn restart_refuses_orphaned_immutable_dependencies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        {
            let store = GraphStore::new(Some(path)).unwrap();
            let ids = resolve(
                &store,
                vec![snapshot("p", "p", &[]), snapshot("c", "c", &["p"])],
            );
            store.rocks.delete_message(&scope(), &ids[0]).unwrap();
        }
        assert!(matches!(
            GraphStore::new(Some(path)),
            Err(GraphError::ImmutableViolation(_))
        ));
    }
}

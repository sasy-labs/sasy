//! Independent compiler/runtime differentials. Included as a child of
//! compiler_tests so both backends use the existing authenticated asset helpers.
use super::*;
use crate::engine::GraphUpdate;
use crate::evaluator::types::{EvalAction, EvalAuthRequest, EvalAuthResponse};
use crate::evaluator::Evaluator;

const POLICY: &str = include_str!("compiler_inference_fixture.dl");

fn request(tools: &[&str], session: &str) -> EvalAuthRequest {
    EvalAuthRequest {
        current_node_ids: vec!["current".into()],
        actions: tools
            .iter()
            .map(|name| EvalAction::ToolCall {
                fn_name: (*name).into(),
                args: "{}".into(),
            })
            .collect(),
        entity: Some("fixture-actor".into()),
        roles: vec![],
        tenant_id: Some("fixture-tenant".into()),
        session_id: Some(session.into()),
        principal: Some("fixture-principal".into()),
        action_metadata: vec![],
    }
}

fn edge(session: &str, source: &str, destination: &str) -> GraphUpdate {
    GraphUpdate::EdgeCreated {
        source: source.into(),
        destination: destination.into(),
        message_index: None,
        proximal: None,
        principal: Some("fixture-principal".into()),
        entity: None,
        session_id: session.into(),
    }
}

fn chain(session: &str) -> Vec<GraphUpdate> {
    vec![
        session_node(session, "danger"),
        session_node(session, "middle"),
        session_node(session, "current"),
        edge(session, "danger", "middle"),
        edge(session, "middle", "current"),
    ]
}

/// Compare decisions separately from explanatory attribution, which is additive.
fn decision(response: &EvalAuthResponse) -> serde_json::Value {
    serde_json::Value::Array(
        response
            .results
            .iter()
            .map(|r| {
                let mut reasons: Vec<_> = r
                    .denial_reasons
                    .iter()
                    .map(|d| (d.kind.clone(), d.reason.clone(), d.suggestion.clone()))
                    .collect();
                reasons.sort();
                let mut transforms = r.transform_ids.clone();
                transforms.sort();
                serde_json::json!({"index":r.index,"authorized":r.authorized,
            "authenticated":r.is_authenticated,"denylisted":r.is_denylisted,
            "allowlisted":r.is_allowlisted,"transforms":transforms,
            "deny_if_unauthorized":r.deny_if_unauthorized,
            "allow_passthrough":r.allow_passthrough,
            "requires_approval":r.requires_approval,"reasons":reasons})
            })
            .collect(),
    )
}

async fn assert_pair(
    base: &dyn Evaluator,
    auto: &dyn Evaluator,
    tools: &[&str],
    session: &str,
    allowed: &[bool],
    transform: bool,
) {
    let expected = base
        .query(request(tools, session))
        .await
        .expect("baseline query");
    let actual = auto
        .query(request(tools, session))
        .await
        .expect("automatic query");
    assert_eq!(
        decision(&actual),
        decision(&expected),
        "tools={tools:?} session={session}"
    );
    assert_eq!(
        actual
            .results
            .iter()
            .map(|r| r.authorized)
            .collect::<Vec<_>>(),
        allowed,
        "an identical bug in both compiles must not certify the differential"
    );
    for (index, result) in actual.results.iter().enumerate() {
        assert_eq!(result.index as usize, index);
        assert_eq!(
            result
                .transform_ids
                .iter()
                .any(|id| id == "fixture-transform"),
            transform && tools[index] == "transform"
        );
    }
}

async fn updates(base: &dyn Evaluator, auto: &dyn Evaluator, values: Vec<GraphUpdate>) {
    base.update(values.clone()).await.expect("baseline update");
    auto.update(values).await.expect("automatic update");
}

fn make(backend: Backend, source: &str, dir: &Path) -> std::sync::Arc<dyn Evaluator> {
    if backend == Backend::Compiled {
        let built = compile_souffle_with_assets(source, None, dir, &discover_for_test()).unwrap();
        std::sync::Arc::new(
            crate::evaluator::manager::EvaluatorProcess::souffle(
                built.binary_path,
                "policy_program".into(),
            )
            .unwrap(),
        )
    } else {
        let path = dir.join("fixture.dl");
        std::fs::write(&path, source).unwrap();
        let built = crate::evaluator::factory::build_souffle_factory_with_artifacts(
            sasy_common::Backend::SouffleInterpreted,
            dir,
            &path,
            None,
            Some(&interpreted_assets_for_test()),
        )
        .expect("interpreted assets required; no silent skip");
        (built.factory)().expect("interpreted spawn")
    }
}

async fn differential(backend: Backend) {
    let baseline_dir = TempDir::new().unwrap();
    let automatic_dir = TempDir::new().unwrap();
    // Explicit author gates must remain authoritative. This is the independent
    // baseline; no test-only environment switch or alternative compiler exists.
    let baseline =
        format!("{POLICY}\nCurrentDependsPolicyRelevant().\nReachableFromPolicyRelevant().\n");
    let base = make(backend, &baseline, baseline_dir.path());
    let auto = make(backend, POLICY, automatic_dir.path());
    let (base, auto) = (base.as_ref(), auto.as_ref());

    assert_pair(
        base,
        auto,
        &["read", "write", "export", "audit", "transform"],
        "s-a",
        &[true, true, true, false, true],
        false,
    )
    .await;
    // Keep Actions fixed while a nonrecursive state prerequisite changes. These
    // updates must be retained even when inference suppresses recursive work.
    assert_pair(base, auto, &["state-write"], "s-state", &[true], false).await;
    updates(base, auto, chain("s-state")).await;
    assert_pair(base, auto, &["state-write"], "s-state", &[true], false).await;
    updates(base, auto, vec![session_node("s-state", "state-signal")]).await;
    assert_pair(base, auto, &["state-write"], "s-state", &[false], false).await;
    assert_pair(
        base,
        auto,
        &["read", "state-write"],
        "s-state",
        &[true, false],
        false,
    )
    .await;
    assert_pair(
        base,
        auto,
        &["state-write", "read"],
        "s-state",
        &[false, true],
        false,
    )
    .await;
    updates(
        base,
        auto,
        vec![GraphUpdate::NodeDeleted("state-signal".into())],
    )
    .await;
    assert_pair(base, auto, &["state-write"], "s-state", &[true], false).await;
    updates(
        base,
        auto,
        vec![GraphUpdate::EdgeDeleted {
            source: "danger".into(),
            destination: "middle".into(),
        }],
    )
    .await;
    updates(base, auto, vec![session_node("s-state", "state-signal")]).await;
    assert_pair(base, auto, &["state-write"], "s-state", &[true], false).await;
    updates(
        base,
        auto,
        vec![GraphUpdate::NodeDeleted("state-signal".into())],
    )
    .await;
    updates(base, auto, vec![edge("s-state", "danger", "middle")]).await;
    assert_pair(base, auto, &["state-write"], "s-state", &[true], false).await;
    updates(base, auto, vec![session_node("s-state", "state-signal")]).await;
    assert_pair(base, auto, &["state-write"], "s-state", &[false], false).await;
    // Global node IDs above overlap the next session's fixture intentionally;
    // reset before replay to keep the original session-isolation checks exact.
    base.reset().await.unwrap();
    auto.reset().await.unwrap();
    // First run a gate-irrelevant query, then persist graph updates while off.
    assert_pair(base, auto, &["read"], "s-a", &[true], false).await;
    updates(base, auto, chain("s-a")).await;
    assert_pair(base, auto, &["read"], "s-a", &[true], false).await;
    assert_pair(
        base,
        auto,
        &["read", "write", "export", "audit", "transform"],
        "s-a",
        &[true, false, false, false, true],
        true,
    )
    .await;
    assert_pair(
        base,
        auto,
        &["export", "read", "write"],
        "s-a",
        &[false, true, false],
        false,
    )
    .await;
    // A singleton following a mixed batch cannot inherit action demand/results.
    assert_pair(base, auto, &["read"], "s-a", &[true], false).await;
    updates(
        base,
        auto,
        vec![
            session_node("s-a", "approval"),
            edge("s-a", "approval", "current"),
        ],
    )
    .await;
    assert_pair(
        base,
        auto,
        &["audit", "write"],
        "s-a",
        &[true, false],
        false,
    )
    .await;
    updates(
        base,
        auto,
        vec![GraphUpdate::EdgeDeleted {
            source: "danger".into(),
            destination: "middle".into(),
        }],
    )
    .await;
    assert_pair(
        base,
        auto,
        &["write", "export", "audit", "transform"],
        "s-a",
        &[true, true, true, true],
        false,
    )
    .await;
    assert_pair(base, auto, &["read"], "s-a", &[true], false).await;
    updates(base, auto, vec![edge("s-a", "danger", "middle")]).await;
    assert_pair(
        base,
        auto,
        &["write", "transform"],
        "s-a",
        &[false, true],
        true,
    )
    .await;
    assert_pair(
        base,
        auto,
        &["write", "export", "audit"],
        "s-b",
        &[true, true, false],
        false,
    )
    .await;
    base.reset().await.unwrap();
    auto.reset().await.unwrap();
    assert_pair(
        base,
        auto,
        &["write", "export", "audit", "transform"],
        "s-a",
        &[true, true, false, true],
        false,
    )
    .await;
    updates(base, auto, chain("s-a")).await;
    assert_pair(
        base,
        auto,
        &["write", "export"],
        "s-a",
        &[false, false],
        false,
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires policy C++ toolchain; root-owned integration build"]
async fn inferred_gates_match_authored_unconditional_compiled() {
    differential(Backend::Compiled).await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires built interpreted adapter; root-owned integration build"]
async fn inferred_gates_match_authored_unconditional_interpreted() {
    differential(Backend::Interpreted).await;
}

fn diagnostic_status(response: &EvalAuthResponse, index: usize, message: &str) -> String {
    // JSON keeps this independent fixture decoupled from the new struct layout.
    let value = serde_json::to_value(&response.results[index]).unwrap();
    let routes = value["allow_routes"]
        .as_array()
        .expect("runtime must carry attribution rows");
    let matching: Vec<_> = routes.iter().filter(|r| r["details"] == message).collect();
    assert_eq!(
        matching.len(),
        1,
        "one diagnostic for {message}: {routes:?}"
    );
    assert!(!matching[0]["rule_id"].as_str().unwrap().is_empty());
    assert!(!matching[0]["source_location"].as_str().unwrap().is_empty());
    matching[0]["status"].as_str().unwrap().to_string()
}

async fn attribution_cases(backend: Backend) {
    use crate::evaluator::types::{ActionMetadataEntry, PolicyMetadataFact};
    let dir = TempDir::new().unwrap();
    let ev = make(
        backend,
        include_str!("compiler_allow_attribution_fixture.dl"),
        dir.path(),
    );
    let approval = |principal: Option<&str>, roles: Vec<String>, tool: &str| {
        let mut req = request(&[tool], "s-a");
        req.principal = principal.map(str::to_string);
        req.roles = roles;
        req
    };
    for (principal, roles, tool, status) in [
        (None, vec![], "approve", "blocked"),
        (Some("other"), vec!["approver".into()], "approve", "blocked"),
        (Some("trusted"), vec![], "approve", "blocked"),
        (Some("trusted"), vec!["approver".into()], "read", "blocked"),
        (
            Some("trusted"),
            vec!["approver".into()],
            "approve",
            "possible",
        ),
    ] {
        let result = ev.query(approval(principal, roles, tool)).await.unwrap();
        assert!(
            !result.results[0].authorized && !result.results[0].is_allowlisted,
            "a possible explanation is not an actual allow decision"
        );
        assert_eq!(
            diagnostic_status(&result, 0, "fixture-approval-route"),
            status
        );
    }
    let correlated = |right: &str| {
        let mut req = request(&["correlated", "read"], "s-a");
        req.action_metadata = vec![ActionMetadataEntry {
            index: 0,
            facts: vec![
                PolicyMetadataFact {
                    rel: "left".into(),
                    a: "w1".into(),
                    b: "yes".into(),
                },
                PolicyMetadataFact {
                    rel: "right".into(),
                    a: right.into(),
                    b: "yes".into(),
                },
            ],
        }];
        req
    };
    let mismatch = ev.query(correlated("w2")).await.unwrap();
    assert_eq!(
        diagnostic_status(&mismatch, 0, "fixture-correlated-route"),
        "blocked"
    );
    let matched = ev.query(correlated("w1")).await.unwrap();
    assert_eq!(
        diagnostic_status(&matched, 0, "fixture-correlated-route"),
        "possible"
    );
    assert_eq!(
        diagnostic_status(&matched, 1, "fixture-correlated-route"),
        "blocked"
    );
    assert!(!matched.results[0].authorized);
    ev.update(vec![
        session_node("s-a", "approval"),
        session_node("s-a", "current"),
        edge("s-a", "approval", "current"),
    ])
    .await
    .unwrap();
    let approved = ev
        .query(approval(
            Some("trusted"),
            vec!["approver".into()],
            "approve",
        ))
        .await
        .unwrap();
    assert!(approved.results[0].authorized && approved.results[0].is_allowlisted);
    let approved_correlated = ev.query(correlated("w1")).await.unwrap();
    assert!(approved_correlated.results[0].authorized);
    assert!(!approved_correlated.results[1].authorized);
    let still_mismatched = ev.query(correlated("w2")).await.unwrap();
    assert!(!still_mismatched.results[0].authorized);
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires policy C++ toolchain; root-owned integration build"]
async fn allow_attribution_preserves_decisions_and_witnesses_compiled() {
    attribution_cases(Backend::Compiled).await;
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires built interpreted adapter; root-owned integration build"]
async fn allow_attribution_preserves_decisions_and_witnesses_interpreted() {
    attribution_cases(Backend::Interpreted).await;
}

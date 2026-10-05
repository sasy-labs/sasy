//! Integration test: build the cdylib and load it via
//! PluginLoader, exercising the full C ABI round-trip.
//!
//! Invokes `cargo build -p sasy-policy-plugin` to produce
//! the shared library, then loads it through PluginLoader
//! and exercises every PluginEngine method.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use policy_sdk::proto;
use policy_sdk::PluginEngine;
use policy_sdk::PluginLoader;

/// Build the cdylib and return the path to the .so/.dylib.
fn build_cdylib() -> PathBuf {
    // Navigate to the workspace root (two levels up from
    // this crate's manifest dir).
    let workspace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

    let output = Command::new("cargo")
        .args(["build", "-p", "sasy-policy-plugin", "--message-format=json"])
        .current_dir(&workspace_dir)
        .output()
        .expect("failed to run cargo build");

    assert!(
        output.status.success(),
        "cargo build failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Find the cdylib artifact in cargo's JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
                && msg
                    .get("target")
                    .and_then(|t| t.get("name"))
                    .and_then(|n| n.as_str())
                    == Some("sasy_policy_plugin")
            {
                if let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array()) {
                    for f in filenames {
                        let p = f.as_str().unwrap();
                        if p.ends_with(".so") || p.ends_with(".dylib") {
                            return PathBuf::from(p);
                        }
                    }
                }
            }
        }
    }

    panic!("could not find cdylib artifact in cargo output");
}

#[test]
fn plugin_load_and_destroy() {
    let so_path = build_cdylib();
    let loader = PluginLoader::load(&so_path).expect("failed to load plugin");
    let loader = Arc::new(loader);
    let engine = PluginEngine::new(loader).expect("failed to create engine");
    drop(engine);
}

#[test]
fn plugin_check_authorization_empty_graph() {
    let so_path = build_cdylib();
    let loader = Arc::new(PluginLoader::load(&so_path).unwrap());
    let engine = PluginEngine::new(loader).unwrap();

    use sasy_common::policy_engine::{action::ActionType, Action, ToolCallAction};

    let action = Action {
        action_type: Some(ActionType::ToolCall(ToolCallAction {
            fn_name: "web_search".into(),
            args: r#"{"q":"test"}"#.into(),
        })),
        ..Default::default()
    };

    let resp = engine
        .check_authorization(
            &[],
            &[action],
            Some("user1"),
            &["researcher".into()],
            None,
            None,
            None,
        )
        .expect("check_authorization failed");

    assert_eq!(resp.results.len(), 1);
    assert!(
        resp.results[0].authorized,
        "expected authorized with empty graph"
    );
}

#[test]
fn plugin_graph_updates_and_sync_status() {
    let so_path = build_cdylib();
    let loader = Arc::new(PluginLoader::load(&so_path).unwrap());
    let engine = PluginEngine::new(loader).unwrap();

    // Initial status: empty
    let status = engine.get_sync_status().expect("get_sync_status");
    assert_eq!(status.node_count, 0);
    assert_eq!(status.edge_count, 0);

    // Add a node and an edge
    let updates = vec![
        proto::GraphUpdate::NodeCreated {
            id: "n1".into(),
            content: Some("test message".into()),
            role: Some("user".into()),
            agent: Some("TestAgent".into()),
            tools: vec![("tool1".into(), "{}".into())],
            entity: Some("entity1".into()),
            derived_from: None,
            session_id: None,
        },
        proto::GraphUpdate::EdgeCreated {
            source: "n1".into(),
            destination: "n2".into(),
            session_id: None,
            principal: None,
            entity: None,
        },
    ];
    engine
        .apply_graph_updates(updates)
        .expect("apply_graph_updates");

    let status = engine
        .get_sync_status()
        .expect("get_sync_status after updates");
    assert_eq!(status.node_count, 1);
    assert_eq!(status.edge_count, 1);
}

#[test]
fn plugin_reset_clears_state() {
    let so_path = build_cdylib();
    let loader = Arc::new(PluginLoader::load(&so_path).unwrap());
    let engine = PluginEngine::new(loader).unwrap();

    engine
        .apply_graph_updates(vec![proto::GraphUpdate::NodeCreated {
            id: "n1".into(),
            content: None,
            role: None,
            agent: None,
            tools: vec![],
            entity: None,
            derived_from: None,
            session_id: None,
        }])
        .unwrap();

    let status = engine.get_sync_status().unwrap();
    assert_eq!(status.node_count, 1);

    engine.reset().expect("reset");

    let status = engine.get_sync_status().unwrap();
    assert_eq!(status.node_count, 0);
    assert_eq!(status.edge_count, 0);
}

#[test]
fn plugin_set_connected_and_sequence() {
    let so_path = build_cdylib();
    let loader = Arc::new(PluginLoader::load(&so_path).unwrap());
    let engine = PluginEngine::new(loader).unwrap();

    engine.set_connected(true);
    engine.set_sequence(99);

    let status = engine.get_sync_status().unwrap();
    assert!(status.connected);
    assert_eq!(status.current_sequence, 99);

    engine.set_connected(false);
    let status = engine.get_sync_status().unwrap();
    assert!(!status.connected);
}

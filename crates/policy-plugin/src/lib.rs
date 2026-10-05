//! Example policy plugin cdylib.
//!
//! Reference implementation of the C ABI defined by
//! `sasy-policy-sdk`: compiles to a `.so`/`.dylib` that
//! `PluginLoader` can load at runtime. It wraps
//! [`StubEngine`] (allow-all + graph-state tracking) so the
//! plugin loader has a working fixture without pulling in a
//! full policy backend. An embedder building a real plugin
//! swaps `StubEngine` for their own [`Engine`] implementation.

use std::sync::Arc;

use policy_sdk::proto;
use sasy_common::policy_plugin::graph_update_entry::UpdateType;
use sasy_policy::engine::{Engine, GraphUpdate, ToolInfo};
use sasy_policy::StubEngine;

/// Initialize tracing subscriber (once).
fn ensure_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .try_init();
    });
}

/// Opaque engine wrapper stored behind the `void*` handle.
struct PluginHandle {
    engine: Arc<StubEngine>,
    /// Tokio runtime so a real engine swapped in here can use
    /// `Handle::current()` / `block_in_place` in
    /// `check_authorization`. Unused by `StubEngine` itself.
    #[allow(dead_code)]
    runtime: tokio::runtime::Runtime,
}

// ── Lifecycle ──────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sasy_policy_create() -> *mut std::ffi::c_void {
    ensure_tracing();

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => return std::ptr::null_mut(),
    };

    let engine = StubEngine::new();

    let handle = Box::new(PluginHandle {
        engine,
        runtime: rt,
    });
    Box::into_raw(handle) as *mut std::ffi::c_void
}

/// # Safety
///
/// `engine` must be a handle previously returned by
/// `sasy_policy_create` (or null). It must not be used again after
/// this call; passing the same handle twice is a double free.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_destroy(engine: *mut std::ffi::c_void) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine as *mut PluginHandle);
    }
}

// ── Authorization ──────────────────────────────────────

/// # Safety
///
/// `engine` must be a valid, non-null handle from
/// `sasy_policy_create`. `request_buf`/`request_len` must describe a
/// valid readable region holding an encoded request. On success the
/// callee writes a freshly allocated buffer into `*response_buf` and
/// its length into `*response_len`; the caller must free it with
/// `sasy_policy_free`. Both out-pointers must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_check_authorization(
    engine: *mut std::ffi::c_void,
    request_buf: *const u8,
    request_len: usize,
    response_buf: *mut *mut u8,
    response_len: *mut usize,
) -> i32 {
    let handle = &*(engine as *const PluginHandle);

    let req_slice = std::slice::from_raw_parts(request_buf, request_len);
    let req = match proto::decode_check_auth_request(req_slice) {
        Ok(r) => r,
        Err(_) => return 1,
    };

    let entity_ref = req.entity.as_deref();
    // Enter the Tokio runtime context so that an async engine
    // swapped in here can use Handle::current() + block_in_place.
    let _guard = handle.runtime.enter();
    // Plugin SDK has no per-request auth context, so all plugin
    // traffic lands in the host's `default` tenant global partition.
    // Multi-tenant plugin embedders need to thread tenant through
    // their own ABI extension.
    let scope = sasy_common::SessionScope::global("default");
    // The plugin SDK does not carry a server-stamped principal, so
    // `None` is passed at this boundary; extending the plugin
    // protocol would let embedders carry their own principal.
    let result = handle.engine.check_authorization(
        &req.current_node_ids,
        &req.actions,
        entity_ref,
        &req.roles,
        &scope,
        None,
        None,
    );

    match result {
        Ok(resp) => {
            let buf = proto::encode_authorization_response(&resp);
            let len = buf.len();
            let ptr = alloc_buf(buf);
            *response_buf = ptr;
            *response_len = len;
            0
        }
        Err(_) => 2,
    }
}

// ── Graph updates ──────────────────────────────────────

/// # Safety
///
/// `engine` must be a valid, non-null handle from
/// `sasy_policy_create`. `updates_buf`/`updates_len` must describe a
/// valid readable region holding an encoded graph-update batch.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_apply_graph_updates(
    engine: *mut std::ffi::c_void,
    updates_buf: *const u8,
    updates_len: usize,
) -> i32 {
    let handle = &*(engine as *const PluginHandle);

    let slice = std::slice::from_raw_parts(updates_buf, updates_len);
    let batch = match proto::decode_graph_update_batch(slice) {
        Ok(b) => b,
        Err(_) => return 1,
    };

    let updates: Vec<GraphUpdate> = batch
        .updates
        .into_iter()
        .filter_map(|entry| entry.update_type.map(proto_to_graph_update))
        .collect();

    match handle.engine.apply_graph_updates(updates) {
        Ok(()) => 0,
        Err(_) => 2,
    }
}

/// Convert a proto `UpdateType` to an engine `GraphUpdate`.
fn proto_to_graph_update(update: UpdateType) -> GraphUpdate {
    match update {
        UpdateType::NodeCreated(node) => GraphUpdate::NodeCreated {
            id: node.id,
            content: node.content,
            role: node.role,
            agent: node.agent,
            tools: node
                .tools
                .into_iter()
                .map(|t| ToolInfo {
                    name: t.name,
                    arguments: t.arguments,
                })
                .collect(),
            entity: node.entity,
            // Plugin SDK proto doesn't carry an auth-derived
            // principal; the host that loads the plugin sees the
            // gRPC auth context separately.
            principal: None,
            derived_from: node.derived_from.map(|t| ToolInfo {
                name: t.name,
                arguments: t.arguments,
            }),
            // The plugin proto carries no message metadata either.
            metadata: None,
            session_id: String::new(),
        },
        UpdateType::NodeDeleted(id) => GraphUpdate::NodeDeleted(id),
        UpdateType::EdgeCreated(edge) => GraphUpdate::EdgeCreated {
            source: edge.source,
            destination: edge.destination,
            // Plugin SDK proto doesn't yet expose these fields.
            message_index: None,
            proximal: None,
            principal: edge.principal,
            entity: edge.entity,
            session_id: String::new(),
        },
        UpdateType::EdgeDeleted(edge) => GraphUpdate::EdgeDeleted {
            source: edge.source,
            destination: edge.destination,
        },
    }
}

// ── State management ───────────────────────────────────

/// # Safety
///
/// `engine` must be a valid, non-null handle previously returned by
/// `sasy_policy_create`.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_reset(engine: *mut std::ffi::c_void) -> i32 {
    let handle = &*(engine as *const PluginHandle);
    match handle.engine.reset() {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// # Safety
///
/// `engine` must be a valid, non-null handle from
/// `sasy_policy_create`. `path`/`path_len` must describe a valid
/// readable region holding a UTF-8 filesystem path.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_load_rule_metadata(
    engine: *mut std::ffi::c_void,
    path: *const u8,
    path_len: usize,
) -> i32 {
    let handle = &*(engine as *const PluginHandle);

    let path_str = match std::str::from_utf8(std::slice::from_raw_parts(path, path_len)) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    let path = std::path::Path::new(path_str);

    match handle.engine.load_rule_metadata(path) {
        Ok(()) => 0,
        Err(_) => 2,
    }
}

// ── Sync status ────────────────────────────────────────

/// # Safety
///
/// `engine` must be a valid, non-null handle from
/// `sasy_policy_create`. On success the callee writes a freshly
/// allocated buffer into `*status_buf` and its length into
/// `*status_len`; the caller must free it with `sasy_policy_free`.
/// Both out-pointers must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_get_sync_status(
    engine: *mut std::ffi::c_void,
    status_buf: *mut *mut u8,
    status_len: *mut usize,
) -> i32 {
    let handle = &*(engine as *const PluginHandle);
    let status = handle.engine.get_sync_status();

    let proto_status = sasy_common::policy_plugin::SyncStatusProto {
        current_sequence: status.current_sequence,
        node_count: status.node_count as u64,
        edge_count: status.edge_count as u64,
        connected: status.connected,
    };

    let buf = proto::encode_sync_status(&proto_status);
    let len = buf.len();
    let ptr = alloc_buf(buf);
    *status_buf = ptr;
    *status_len = len;
    0
}

/// # Safety
///
/// `engine` must be a valid, non-null handle previously returned by
/// `sasy_policy_create`.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_set_connected(engine: *mut std::ffi::c_void, connected: i32) {
    let handle = &*(engine as *const PluginHandle);
    handle.engine.set_connected(connected != 0);
}

/// # Safety
///
/// `engine` must be a valid, non-null handle previously returned by
/// `sasy_policy_create`.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_set_sequence(engine: *mut std::ffi::c_void, seq: i64) {
    let handle = &*(engine as *const PluginHandle);
    handle.engine.set_sequence(seq);
}

// ── Memory management ──────────────────────────────────

/// # Safety
///
/// `buf`/`len` must exactly match a buffer previously returned by this
/// plugin (e.g. via `sasy_policy_check_authorization` or
/// `sasy_policy_get_sync_status`), or `buf` must be null. The buffer
/// must not be used or freed again after this call.
#[no_mangle]
pub unsafe extern "C" fn sasy_policy_free(buf: *mut u8, len: usize) {
    if !buf.is_null() && len > 0 {
        let _ = Vec::from_raw_parts(buf, len, len);
    }
}

/// Allocate a buffer from a Vec and return a raw pointer.
///
/// The caller must free it with `sasy_policy_free`.
fn alloc_buf(buf: Vec<u8>) -> *mut u8 {
    let mut buf = buf.into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

// ── Version ────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn sasy_policy_abi_version() -> u32 {
    policy_sdk::ABI_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use sasy_common::policy_engine::{action::ActionType, Action, ToolCallAction};

    /// Helper: create + return a raw handle.
    fn create_handle() -> *mut std::ffi::c_void {
        let h = sasy_policy_create();
        assert!(!h.is_null(), "create returned null");
        h
    }

    #[test]
    fn test_create_and_destroy() {
        let h = create_handle();
        unsafe { sasy_policy_destroy(h) };
    }

    #[test]
    fn test_abi_version_matches_sdk() {
        assert_eq!(sasy_policy_abi_version(), policy_sdk::ABI_VERSION,);
    }

    #[test]
    fn test_check_authorization_empty_graph() {
        let h = create_handle();

        let action = Action {
            action_type: Some(ActionType::ToolCall(ToolCallAction {
                fn_name: "search".into(),
                args: "{}".into(),
            })),
            ..Default::default()
        };
        let req = proto::build_check_auth_request(
            &[],
            &[action],
            Some("user1"),
            &["researcher".into()],
            None,
            None,
            None,
        );
        let req_buf = proto::encode_check_auth_request(&req);

        let mut resp_buf: *mut u8 = std::ptr::null_mut();
        let mut resp_len: usize = 0;

        let rc = unsafe {
            sasy_policy_check_authorization(
                h,
                req_buf.as_ptr(),
                req_buf.len(),
                &mut resp_buf,
                &mut resp_len,
            )
        };
        assert_eq!(rc, 0, "check_authorization failed");
        assert!(!resp_buf.is_null());
        assert!(resp_len > 0);

        let resp_slice = unsafe { std::slice::from_raw_parts(resp_buf, resp_len) };
        let resp = proto::decode_authorization_response(resp_slice).expect("decode response");
        assert_eq!(resp.results.len(), 1);

        unsafe { sasy_policy_free(resp_buf, resp_len) };
        unsafe { sasy_policy_destroy(h) };
    }

    #[test]
    fn test_apply_graph_updates_roundtrip() {
        let h = create_handle();

        let updates = vec![
            policy_sdk::proto::GraphUpdate::NodeCreated {
                id: "n1".into(),
                content: Some("hello".into()),
                role: Some("user".into()),
                agent: Some("Agent".into()),
                tools: vec![],
                entity: None,
                derived_from: None,
                session_id: None,
            },
            policy_sdk::proto::GraphUpdate::EdgeCreated {
                source: "n1".into(),
                destination: "n2".into(),
                session_id: None,
                principal: None,
                entity: None,
            },
        ];
        let batch = proto::build_graph_update_batch(&updates);
        let buf = proto::encode_graph_update_batch(&batch);

        let rc = unsafe { sasy_policy_apply_graph_updates(h, buf.as_ptr(), buf.len()) };
        assert_eq!(rc, 0, "apply_graph_updates failed");

        unsafe { sasy_policy_destroy(h) };
    }

    #[test]
    fn test_sync_status_after_updates() {
        let h = create_handle();

        // Add a node and an edge
        let updates = vec![
            policy_sdk::proto::GraphUpdate::NodeCreated {
                id: "n1".into(),
                content: None,
                role: None,
                agent: None,
                tools: vec![],
                entity: None,
                derived_from: None,
                session_id: None,
            },
            policy_sdk::proto::GraphUpdate::EdgeCreated {
                source: "n1".into(),
                destination: "n2".into(),
                session_id: None,
                principal: None,
                entity: None,
            },
        ];
        let batch = proto::build_graph_update_batch(&updates);
        let buf = proto::encode_graph_update_batch(&batch);
        let rc = unsafe { sasy_policy_apply_graph_updates(h, buf.as_ptr(), buf.len()) };
        assert_eq!(rc, 0);

        // Check sync status
        let mut status_buf: *mut u8 = std::ptr::null_mut();
        let mut status_len: usize = 0;
        let rc = unsafe { sasy_policy_get_sync_status(h, &mut status_buf, &mut status_len) };
        assert_eq!(rc, 0);

        let slice = unsafe { std::slice::from_raw_parts(status_buf, status_len) };
        let status = proto::decode_sync_status(slice).expect("decode status");
        assert_eq!(status.node_count, 1);
        assert_eq!(status.edge_count, 1);

        unsafe {
            sasy_policy_free(status_buf, status_len);
        }

        // Reset and verify counts go back to 0
        let rc = unsafe { sasy_policy_reset(h) };
        assert_eq!(rc, 0);

        let mut status_buf2: *mut u8 = std::ptr::null_mut();
        let mut status_len2: usize = 0;
        let rc = unsafe { sasy_policy_get_sync_status(h, &mut status_buf2, &mut status_len2) };
        assert_eq!(rc, 0);

        let slice2 = unsafe { std::slice::from_raw_parts(status_buf2, status_len2) };
        let status2 = proto::decode_sync_status(slice2).expect("decode status after reset");
        assert_eq!(status2.node_count, 0);
        assert_eq!(status2.edge_count, 0);

        unsafe {
            sasy_policy_free(status_buf2, status_len2);
        }
        unsafe { sasy_policy_destroy(h) };
    }

    #[test]
    fn test_set_connected_and_sequence() {
        let h = create_handle();

        unsafe { sasy_policy_set_connected(h, 1) };
        unsafe { sasy_policy_set_sequence(h, 42) };

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        let rc = unsafe { sasy_policy_get_sync_status(h, &mut buf, &mut len) };
        assert_eq!(rc, 0);

        let slice = unsafe { std::slice::from_raw_parts(buf, len) };
        let status = proto::decode_sync_status(slice).expect("decode status");
        assert!(status.connected);
        assert_eq!(status.current_sequence, 42);

        unsafe { sasy_policy_free(buf, len) };
        unsafe { sasy_policy_destroy(h) };
    }
}

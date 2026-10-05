//! SDK for loading sasy policy plugins via stable C ABI.
//!
//! A policy plugin is a shared library (`.so` on Linux,
//! `.dylib` on macOS) that exports a set of C functions
//! implementing the policy engine contract. This crate
//! provides:
//!
//! - [`ABI_VERSION`] — the version both sides must agree
//!   on
//! - [`PluginLoader`] — dlopen wrapper that resolves all
//!   required symbols
//! - [`PluginEngine`] — wraps a loaded plugin and
//!   implements `sasy_policy::engine::Engine`
//! - [`abi`] — C function type signatures
//! - [`proto`] — protobuf serialization helpers

pub mod abi;
pub mod loader;
pub mod proto;

pub use loader::PluginLoader;

use std::ptr;
use std::sync::Arc;

use anyhow::{bail, Result};
use sasy_common::policy_engine::{Action, AuthorizationResponse};
use sasy_common::policy_plugin::SyncStatusProto;
use tracing::info;

/// ABI version. Both the SDK (loader) and the plugin
/// must agree on this value.
pub const ABI_VERSION: u32 = 1;

/// A policy engine backed by a dynamically loaded plugin.
///
/// Implements the same contract as `sasy_policy::Engine`
/// by serializing arguments to protobuf, calling the
/// plugin's C functions, and deserializing responses.
///
/// The plugin handle is created on construction and
/// destroyed on drop.
pub struct PluginEngine {
    loader: Arc<PluginLoader>,
    handle: *mut std::ffi::c_void,
}

// SAFETY: The plugin's C functions use internal locking
// (the wrapped engine has Mutex/RwLock). The opaque handle
// is only accessed through those thread-safe functions.
unsafe impl Send for PluginEngine {}
unsafe impl Sync for PluginEngine {}

impl PluginEngine {
    /// Create a new plugin engine from a loaded plugin.
    ///
    /// Calls `sasy_policy_create()` to obtain an opaque
    /// engine handle.
    pub fn new(loader: Arc<PluginLoader>) -> Result<Self> {
        let handle = unsafe { (loader.create)() };
        if handle.is_null() {
            bail!("sasy_policy_create returned null");
        }
        info!("plugin engine created");
        Ok(Self { loader, handle })
    }

    /// Free a buffer allocated by the plugin.
    unsafe fn free_plugin_buf(&self, buf: *mut u8, len: usize) {
        (self.loader.free)(buf, len);
    }

    /// Apply graph updates via the C ABI.
    pub fn apply_graph_updates(&self, updates: Vec<proto::GraphUpdate>) -> Result<()> {
        let batch = proto::build_graph_update_batch(&updates);
        let buf = proto::encode_graph_update_batch(&batch);

        let rc = unsafe { (self.loader.apply_graph_updates)(self.handle, buf.as_ptr(), buf.len()) };
        if rc != 0 {
            bail!(
                "sasy_policy_apply_graph_updates \
                 returned error code {}",
                rc
            );
        }
        Ok(())
    }

    /// Check authorization via the C ABI.
    #[allow(clippy::too_many_arguments)] // inherent: mirrors the C-ABI authorization signature
    pub fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: &[Action],
        entity: Option<&str>,
        roles: &[String],
        session_id: Option<&str>,
        tenant: Option<&str>,
        principal: Option<&str>,
    ) -> Result<AuthorizationResponse> {
        let req = proto::build_check_auth_request(
            current_node_ids,
            actions,
            entity,
            roles,
            session_id,
            tenant,
            principal,
        );
        let req_buf = proto::encode_check_auth_request(&req);

        let mut resp_buf: *mut u8 = ptr::null_mut();
        let mut resp_len: usize = 0;

        let rc = unsafe {
            (self.loader.check_authorization)(
                self.handle,
                req_buf.as_ptr(),
                req_buf.len(),
                &mut resp_buf,
                &mut resp_len,
            )
        };
        if rc != 0 {
            bail!(
                "sasy_policy_check_authorization \
                 returned error code {}",
                rc
            );
        }

        let resp_slice = unsafe { std::slice::from_raw_parts(resp_buf, resp_len) };
        let response = proto::decode_authorization_response(resp_slice)?;

        unsafe {
            self.free_plugin_buf(resp_buf, resp_len);
        }

        Ok(response)
    }

    /// Reset all state.
    pub fn reset(&self) -> Result<()> {
        let rc = unsafe { (self.loader.reset)(self.handle) };
        if rc != 0 {
            bail!(
                "sasy_policy_reset returned error \
                 code {}",
                rc
            );
        }
        Ok(())
    }

    /// Load rule metadata from a `.dl` file path.
    pub fn load_rule_metadata(&self, path: &std::path::Path) -> Result<()> {
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path"))?;
        let rc = unsafe {
            (self.loader.load_rule_metadata)(self.handle, path_str.as_ptr(), path_str.len())
        };
        if rc != 0 {
            bail!(
                "sasy_policy_load_rule_metadata \
                 returned error code {}",
                rc
            );
        }
        Ok(())
    }

    /// Get sync status.
    pub fn get_sync_status(&self) -> Result<SyncStatusProto> {
        let mut buf: *mut u8 = ptr::null_mut();
        let mut len: usize = 0;

        let rc = unsafe { (self.loader.get_sync_status)(self.handle, &mut buf, &mut len) };
        if rc != 0 {
            bail!(
                "sasy_policy_get_sync_status returned \
                 error code {}",
                rc
            );
        }

        let slice = unsafe { std::slice::from_raw_parts(buf, len) };
        let status = proto::decode_sync_status(slice)?;

        unsafe {
            self.free_plugin_buf(buf, len);
        }

        Ok(status)
    }

    /// Set connected flag.
    pub fn set_connected(&self, connected: bool) {
        unsafe {
            (self.loader.set_connected)(self.handle, connected as i32);
        }
    }

    /// Set current sequence number.
    pub fn set_sequence(&self, seq: i64) {
        unsafe {
            (self.loader.set_sequence)(self.handle, seq);
        }
    }
}

impl Drop for PluginEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.loader.destroy)(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_version_constant() {
        assert_eq!(ABI_VERSION, 1);
    }

    #[test]
    fn test_check_auth_request_roundtrip() {
        use sasy_common::policy_engine::{action::ActionType, ToolCallAction};

        let action = Action {
            action_type: Some(ActionType::ToolCall(ToolCallAction {
                fn_name: "test_fn".into(),
                args: "{}".into(),
            })),
            ..Default::default()
        };

        let req = proto::build_check_auth_request(
            &["n1".into(), "n2".into()],
            &[action],
            Some("user1"),
            &["admin".into()],
            Some("trial-7"),
            Some("acme"),
            Some("alice"),
        );

        let buf = proto::encode_check_auth_request(&req);
        let decoded = proto::decode_check_auth_request(&buf).unwrap();

        assert_eq!(decoded.current_node_ids, vec!["n1", "n2"]);
        assert_eq!(decoded.actions.len(), 1);
        assert_eq!(decoded.entity.as_deref(), Some("user1"));
        assert_eq!(decoded.roles, vec!["admin"]);
        assert_eq!(decoded.session_id.as_deref(), Some("trial-7"));
        assert_eq!(decoded.tenant.as_deref(), Some("acme"));
        assert_eq!(decoded.principal.as_deref(), Some("alice"));
    }

    #[test]
    fn test_graph_update_batch_roundtrip() {
        let updates = vec![
            proto::GraphUpdate::NodeCreated {
                id: "n1".into(),
                content: Some("hello".into()),
                role: Some("user".into()),
                agent: Some("TestAgent".into()),
                tools: vec![("tool1".into(), "{}".into())],
                entity: None,
                derived_from: None,
                session_id: Some("trial-7".into()),
            },
            proto::GraphUpdate::NodeDeleted("n2".into()),
            proto::GraphUpdate::EdgeCreated {
                source: "n1".into(),
                destination: "n3".into(),
                session_id: Some("trial-7".into()),
                principal: None,
                entity: None,
            },
            proto::GraphUpdate::EdgeDeleted {
                source: "n4".into(),
                destination: "n5".into(),
            },
        ];

        let batch = proto::build_graph_update_batch(&updates);
        let buf = proto::encode_graph_update_batch(&batch);
        let decoded = proto::decode_graph_update_batch(&buf).unwrap();

        assert_eq!(decoded.updates.len(), 4);
    }

    /// An edge crosses the plugin ABI with the principal that
    /// asserted it and the entity the recording client named.
    #[test]
    fn edge_update_carries_its_attribution_over_the_abi() {
        let updates = vec![proto::GraphUpdate::EdgeCreated {
            source: "n1".into(),
            destination: "n2".into(),
            session_id: Some("trial-7".into()),
            principal: Some("gateway".into()),
            entity: Some("ingest-worker".into()),
        }];

        let batch = proto::build_graph_update_batch(&updates);
        let buf = proto::encode_graph_update_batch(&batch);
        let decoded = proto::decode_graph_update_batch(&buf).unwrap();

        let entry = decoded.updates.first().expect("one update");
        use sasy_common::policy_plugin::graph_update_entry::UpdateType;
        match entry.update_type.as_ref().expect("update type") {
            UpdateType::EdgeCreated(edge) => {
                assert_eq!(edge.principal.as_deref(), Some("gateway"));
                assert_eq!(edge.entity.as_deref(), Some("ingest-worker"));
            }
            other => panic!("expected EdgeCreated, got {other:?}"),
        }
    }

    #[test]
    fn test_plugin_loader_missing_so() {
        let result = PluginLoader::load(std::path::Path::new("/nonexistent/libsasy_policy.so"));
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("failed to load plugin"),
            "unexpected error: {}",
            err
        );
    }
}

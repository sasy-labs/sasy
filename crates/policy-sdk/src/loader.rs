//! Dynamic library loader for policy plugins.
//!
//! Uses `libloading` to dlopen a `.so`/`.dylib` and
//! resolve the required C ABI symbols.

use std::path::Path;

use anyhow::{bail, Context, Result};
use libloading::{Library, Symbol};
use tracing::info;

use crate::abi;
use crate::ABI_VERSION;

/// A loaded policy plugin library.
///
/// Holds the `Library` handle (keeping the `.so` mapped)
/// and resolved function pointers.
pub struct PluginLoader {
    // Must live as long as the symbols are used.
    _library: Library,
    pub(crate) create: abi::CreateFn,
    pub(crate) destroy: abi::DestroyFn,
    pub(crate) check_authorization: abi::CheckAuthorizationFn,
    pub(crate) apply_graph_updates: abi::ApplyGraphUpdatesFn,
    pub(crate) reset: abi::ResetFn,
    pub(crate) load_rule_metadata: abi::LoadRuleMetadataFn,
    pub(crate) get_sync_status: abi::GetSyncStatusFn,
    pub(crate) set_connected: abi::SetConnectedFn,
    pub(crate) set_sequence: abi::SetSequenceFn,
    pub(crate) free: abi::FreeFn,
}

// SAFETY: The plugin's C functions are expected to be
// thread-safe (the wrapped engine uses internal
// locking). The Library handle is Send+Sync.
unsafe impl Send for PluginLoader {}
unsafe impl Sync for PluginLoader {}

impl PluginLoader {
    /// Load a policy plugin from a shared library path.
    ///
    /// Resolves all required symbols and checks the ABI
    /// version. Returns an error if the library cannot be
    /// loaded, any symbol is missing, or the ABI version
    /// doesn't match.
    pub fn load(path: &Path) -> Result<Self> {
        info!(
            path = %path.display(),
            "loading policy plugin"
        );

        // SAFETY: Loading a shared library can execute
        // arbitrary code (constructors). We trust the
        // user-provided plugin path.
        let library = unsafe {
            Library::new(path)
                .with_context(|| format!("failed to load plugin: {}", path.display()))?
        };

        // Check ABI version first
        let abi_version: Symbol<abi::AbiVersionFn> = unsafe {
            library
                .get(b"sasy_policy_abi_version")
                .context("missing sasy_policy_abi_version")?
        };
        let version = unsafe { abi_version() };
        if version != ABI_VERSION {
            bail!(
                "ABI version mismatch: plugin has v{}, \
                 SDK expects v{}",
                version,
                ABI_VERSION
            );
        }
        info!(abi_version = version, "ABI version OK");

        // Resolve all symbols
        let create: abi::CreateFn = unsafe {
            *library
                .get::<abi::CreateFn>(b"sasy_policy_create")
                .context("missing sasy_policy_create")?
        };
        let destroy: abi::DestroyFn = unsafe {
            *library
                .get::<abi::DestroyFn>(b"sasy_policy_destroy")
                .context("missing sasy_policy_destroy")?
        };
        let check_authorization: abi::CheckAuthorizationFn = unsafe {
            *library
                .get::<abi::CheckAuthorizationFn>(b"sasy_policy_check_authorization")
                .context(
                    "missing \
                         sasy_policy_check_authorization",
                )?
        };
        let apply_graph_updates: abi::ApplyGraphUpdatesFn = unsafe {
            *library
                .get::<abi::ApplyGraphUpdatesFn>(b"sasy_policy_apply_graph_updates")
                .context(
                    "missing \
                         sasy_policy_apply_graph_updates",
                )?
        };
        let reset: abi::ResetFn = unsafe {
            *library
                .get::<abi::ResetFn>(b"sasy_policy_reset")
                .context("missing sasy_policy_reset")?
        };
        let load_rule_metadata: abi::LoadRuleMetadataFn = unsafe {
            *library
                .get::<abi::LoadRuleMetadataFn>(b"sasy_policy_load_rule_metadata")
                .context(
                    "missing \
                         sasy_policy_load_rule_metadata",
                )?
        };
        let get_sync_status: abi::GetSyncStatusFn = unsafe {
            *library
                .get::<abi::GetSyncStatusFn>(b"sasy_policy_get_sync_status")
                .context(
                    "missing \
                         sasy_policy_get_sync_status",
                )?
        };
        let set_connected: abi::SetConnectedFn = unsafe {
            *library
                .get::<abi::SetConnectedFn>(b"sasy_policy_set_connected")
                .context("missing sasy_policy_set_connected")?
        };
        let set_sequence: abi::SetSequenceFn = unsafe {
            *library
                .get::<abi::SetSequenceFn>(b"sasy_policy_set_sequence")
                .context("missing sasy_policy_set_sequence")?
        };
        let free: abi::FreeFn = unsafe {
            *library
                .get::<abi::FreeFn>(b"sasy_policy_free")
                .context("missing sasy_policy_free")?
        };

        Ok(Self {
            _library: library,
            create,
            destroy,
            check_authorization,
            apply_graph_updates,
            reset,
            load_rule_metadata,
            get_sync_status,
            set_connected,
            set_sequence,
            free,
        })
    }
}

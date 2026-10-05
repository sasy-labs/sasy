//! C ABI function type signatures for the policy plugin.
//!
//! These types describe the `extern "C"` functions that a
//! policy plugin `.so`/`.dylib` must export.

/// Create engine instance, return opaque handle.
pub type CreateFn = unsafe extern "C" fn() -> *mut std::ffi::c_void;

/// Destroy engine instance.
pub type DestroyFn = unsafe extern "C" fn(engine: *mut std::ffi::c_void);

/// Check authorization.
///
/// `request_buf`/`request_len`: serialized
/// `CheckAuthRequest` proto.
///
/// On success, `*response_buf` and `*response_len` are
/// set to a serialized `AuthorizationResponse` proto
/// allocated by the plugin. Caller must free with
/// [`FreeFn`].
///
/// Returns 0 on success, nonzero on error.
pub type CheckAuthorizationFn = unsafe extern "C" fn(
    engine: *mut std::ffi::c_void,
    request_buf: *const u8,
    request_len: usize,
    response_buf: *mut *mut u8,
    response_len: *mut usize,
) -> i32;

/// Apply graph updates.
///
/// `updates_buf`/`updates_len`: serialized
/// `GraphUpdateBatch` proto.
///
/// Returns 0 on success, nonzero on error.
pub type ApplyGraphUpdatesFn = unsafe extern "C" fn(
    engine: *mut std::ffi::c_void,
    updates_buf: *const u8,
    updates_len: usize,
) -> i32;

/// Reset all state.
pub type ResetFn = unsafe extern "C" fn(engine: *mut std::ffi::c_void) -> i32;

/// Load rule metadata from a `.dl` file path.
pub type LoadRuleMetadataFn =
    unsafe extern "C" fn(engine: *mut std::ffi::c_void, path: *const u8, path_len: usize) -> i32;

/// Get sync status.
///
/// On success, `*status_buf` and `*status_len` are set
/// to a serialized `SyncStatusProto` proto allocated by
/// the plugin. Caller must free with [`FreeFn`].
pub type GetSyncStatusFn = unsafe extern "C" fn(
    engine: *mut std::ffi::c_void,
    status_buf: *mut *mut u8,
    status_len: *mut usize,
) -> i32;

/// Set connected flag.
pub type SetConnectedFn = unsafe extern "C" fn(engine: *mut std::ffi::c_void, connected: i32);

/// Set current sequence number.
pub type SetSequenceFn = unsafe extern "C" fn(engine: *mut std::ffi::c_void, seq: i64);

/// Free a buffer allocated by the plugin.
pub type FreeFn = unsafe extern "C" fn(buf: *mut u8, len: usize);

/// Get the ABI version supported by this plugin.
pub type AbiVersionFn = unsafe extern "C" fn() -> u32;

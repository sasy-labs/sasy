//! gRPC server wrapping the `sasy-graph` store.
//!
//! Provides two services:
//! - [`ObservabilityService`] — event/computation CRUD
//!   and graph slicing
//! - [`UpdatesService`] — full-state queries and
//!   bidirectional update streaming

pub mod error;
pub mod observability_service;
pub mod rbac;
pub mod updates_service;

pub use observability_service::ObservabilityService;
pub use updates_service::UpdatesService;

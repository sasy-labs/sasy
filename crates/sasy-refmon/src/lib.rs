//! Reference monitor proxy with policy enforcement.
//!
//! Provides:
//! - gRPC RMProxy service (ProxyHTTP, CheckToolCall)
//! - Transform executor for credential injection
//! - HTTP proxy via reqwest
//! - PolicyChecker trait for in-process or remote policy
//!   engine integration

pub mod error;
#[cfg(feature = "proxy")]
pub mod forward_proxy;
#[cfg(feature = "proxy")]
pub(crate) mod net_guard;
pub mod policy;
pub mod service;
#[cfg(all(test, feature = "proxy"))]
pub(crate) mod test_support;
#[cfg(feature = "proxy")]
pub mod transforms;

pub use error::RefmonError;
pub use policy::PolicyChecker;
pub use service::RefmonService;
#[cfg(feature = "proxy")]
pub use transforms::{Transform, TransformConfig, TransformExecutor};

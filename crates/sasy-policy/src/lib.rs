//! Policy engine with pluggable Datalog backends.
//!
//! This crate provides:
//!
//! - [`Engine`] trait for policy evaluation orchestration
//! - [`Evaluator`] trait for backend-agnostic Datalog evaluation
//!   (Soufflé evaluator subprocesses via IPC)
//! - [`EvaluatorEngine`] wrapping any `Evaluator` with sync
//!   tracking, denial traces, and timing
//! - [`SyncManager`] subscribing to graph updates and feeding
//!   them to the engine
//! - gRPC [`PolicyService`] for external callers

#[cfg(feature = "compiler")]
pub mod assets;
#[cfg(feature = "compiler")]
pub mod compiler;
pub mod engine;
pub mod evaluator;
pub mod evaluator_engine;
pub mod hash;
pub mod latency_log;
#[cfg(feature = "llm")]
pub mod llm;
pub(crate) mod nix_runtime;
pub mod oracle_redaction;
pub mod policy_registry;
pub mod policy_types;
pub mod replay;
#[cfg(any(test, feature = "compiler"))]
pub mod runtime_conformance;
pub mod sandbox;
pub mod seccomp;
pub mod service;
pub mod session_evaluator;
// The build-cache machinery (Soufflé/g++ version probes, compiled-binary cache)
// is used only by the compile path; the restricted build has no compiler, so
// gate it out to keep that build free of toolchain references.
#[cfg(feature = "compiler")]
pub mod souffle_cache;
#[cfg(feature = "compiler")]
pub(crate) mod souffle_include;
pub mod sync;
pub mod trace;

/// Shared lock for tests that change compile environment variables or the
/// working directory. Every such test must use this same lock across modules.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Static analysis + rule-metadata live in the `sasy-policy-analysis`
// crate now. Re-exported so existing `crate::analysis::…` /
// `crate::rule_metadata::…` paths (and external `sasy_policy::analysis`
// users like the policy-analyze bin) keep resolving unchanged.
pub use sasy_policy_analysis::{analysis, rule_metadata};

pub use engine::{Engine, StubEngine};
pub use evaluator::{Evaluator, EvaluatorError};
pub use evaluator_engine::EvaluatorEngine;
pub use service::PolicyService;
pub use sync::SyncManager;

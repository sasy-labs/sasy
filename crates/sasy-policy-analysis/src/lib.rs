//! Static analysis for Soufflé policies.
//!
//! - [`analysis`] — the analysis pipeline: lexer + parser → AST,
//!   the FD/congruence chase, and the contradiction / redundancy /
//!   subsumption / conditional-reachability analyses. The
//!   [`analysis::pipeline`] entry points (`analyze_desugared`,
//!   `analyze_raw`) back the `ValidatePolicy` RPC and the
//!   `policy-analyze` binary.
//! - [`rule_metadata`] — parses `// @deny_message:` / `// @suggestion:`
//!   annotations out of policy source. Used both by the analysis
//!   pipeline and by the runtime engine's denial traces.
//!
//! Depends only on `serde` + `tracing` + `thiserror` — no runtime
//! engine, graph store, or proto types — so it builds and tests
//! independently of the policy evaluation core.

pub mod allow_attribution;
pub mod analysis;
pub mod rule_metadata;

pub mod action_gates;

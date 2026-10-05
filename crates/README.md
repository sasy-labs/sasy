# Engine crates

`sasy serve` is one process built from these crates. A request to check a tool
call travels: `sasy-auth` (who is calling, and with what roles) → `sasy-refmon`
(the reference monitor, the component every check passes through) → `sasy-policy`
(evaluates the session's Datalog policy) → `sasy-graph` (the recorded messages
and the dependencies between them).

"Handles untrusted input" below means the crate parses or stores data that
originates outside the engine — model output, tool arguments and results,
credentials presented by a caller, or a policy file uploaded by a client. Those
crates are where a parsing or isolation bug matters most.

| Crate | What it does | Handles untrusted input |
| --- | --- | --- |
| `sasy-binary` | The `sasy` command: `serve`, the policy tools and the local TLS helper | no |
| `sasy-refmon` | Reference-monitor gRPC service: tool-call and HTTP checks, and credential transforms | yes: parses tool names and arguments |
| `sasy-policy` | Compiles policies with Soufflé and runs one evaluator per session | yes: compiles policy files supplied by clients |
| `sasy-policy-analysis` | Static checks on a policy: contradiction, redundancy, reachability, rule metadata | yes: parses policy files supplied by clients |
| `sasy-server` | gRPC service that records messages and their dependencies | yes: stores recorded message content |
| `sasy-graph` | The graph store (in-memory graph over RocksDB), with an optional background Neo4j mirror | yes: stores recorded message content |
| `sasy-auth` | API key, JWT/OIDC and mutual-TLS providers, and the role checks on each RPC | yes: parses credentials presented by callers |
| `sasy-credential` | Credential storage and lookup used by transforms | no |
| `sasy-redaction` | Removes recognized credentials from strings before the engine sends them out | yes: scans recorded content |
| `sasy-common` | Generated gRPC types and the shared domain types every crate uses | yes: decodes incoming requests |
| `policy-sdk` | Loads in-process policy evaluator plugins through a stable C ABI | no, but it loads native code that runs with the engine's privileges |
| `policy-plugin` | An example plugin built against `policy-sdk` | no |
| `sasy-loadgen` | Synthetic gRPC load generator for benchmarks; not part of the served engine | no |

Each crate's `src/lib.rs` or `src/main.rs` opens with what that crate is
responsible for. The policy relations a policy file may use are declared in
`souffle/common_policy.dl` and documented at
https://docs.sasy.ai/policy-language/.

//! The `sasy` binary: `sasy serve` starts the engine.
//!
//! Runs the Observability, ObservabilityUpdates,
//! PolicyEngine, RMProxy, and CredentialServer gRPC
//! services in one process, sharing `Arc` references
//! instead of inter-service gRPC.

// The restricted build strips several services/backends, leaving their
// imports, CLI args, and full-build helpers legitimately unused.
#![cfg_attr(
    feature = "restricted",
    allow(unused_imports, unused_variables, dead_code)
)]

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use policy_sdk::PluginEngine;
use sasy_auth::{AuthConfig, AuthInterceptor, PassthroughAuthProvider};
#[cfg(feature = "proxy")]
use sasy_common::credential_server::credential_server_server::CredentialServerServer;
use sasy_common::observability::{
    observability_server::ObservabilityServer,
    observability_updates_server::ObservabilityUpdatesServer,
};
use sasy_common::policy_engine::{
    self, policy_engine_server::PolicyEngineServer, AuthorizationResponse,
};
use sasy_common::services::rm_proxy_server::RmProxyServer;
#[cfg(feature = "proxy")]
use sasy_credential::{
    CredentialService, CredentialSource, CredentialStore, MemorySource, OpenBaoAuth, OpenBaoConfig,
    OpenBaoSource, SqliteSource,
};
use sasy_graph::GraphStore;
use sasy_policy::service::UserFunctors;
use sasy_policy::{Engine, PolicyService, StubEngine};
use sasy_refmon::RefmonService;
#[cfg(feature = "proxy")]
use sasy_refmon::{TransformConfig, TransformExecutor};
use sasy_server::{ObservabilityService, UpdatesService};
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "restricted")]
mod restricted;

#[cfg(feature = "mtls")]
mod guard_tls;

#[cfg(feature = "mtls")]
mod local_init;

/// Built-in deny-all policy used when no `--policy` is supplied.
///
/// Acts as a safe bootstrap so the binary can come up without an
/// initial policy file: every action falls through to the
/// `Unauthorized` rule from `common_policy.dl`'s denial-by-default
/// path. The operator replaces it via `SetPolicy(scope=Force)`
/// after start-up.
const BOOTSTRAP_DENY_ALL_POLICY: &str = "// Built-in bootstrap policy — denies every action.\n\
// Replace via SetPolicy(scope=Force) once the binary is up.\n\
// IsAuthorized is intentionally never derived; common_policy's\n\
// denial path produces a clear \"not allowlisted\" reason.\n";

// ── PluginEngineAdapter ──────────────────────────────
// Bridges policy_sdk::PluginEngine → sasy_policy::Engine

struct PluginEngineAdapter {
    plugin: PluginEngine,
}

impl Engine for PluginEngineAdapter {
    fn apply_graph_updates(&self, updates: Vec<sasy_policy::engine::GraphUpdate>) -> Result<()> {
        let sdk_updates: Vec<_> = updates
            .into_iter()
            .map(|u| match u {
                sasy_policy::engine::GraphUpdate::NodeCreated {
                    id,
                    content,
                    role,
                    agent,
                    tools,
                    entity,
                    principal: _, // plugin SDK doesn't yet expose principal
                    // The plugin ABI's fact schema carries no message
                    // metadata, so `MessageMetadata` is empty under a
                    // plugin backend and the value stops here.
                    metadata: _,
                    derived_from,
                    session_id,
                } => policy_sdk::proto::GraphUpdate::NodeCreated {
                    id,
                    content,
                    role,
                    agent,
                    tools: tools.into_iter().map(|t| (t.name, t.arguments)).collect(),
                    entity,
                    derived_from: derived_from.map(|t| (t.name, t.arguments)),
                    session_id: if session_id.is_empty() {
                        None
                    } else {
                        Some(session_id)
                    },
                },
                sasy_policy::engine::GraphUpdate::NodeDeleted(id) => {
                    policy_sdk::proto::GraphUpdate::NodeDeleted(id)
                }
                sasy_policy::engine::GraphUpdate::EdgeCreated {
                    source,
                    destination,
                    // The plugin ABI does not carry proximal and
                    // message_index, so they stop at this boundary.
                    message_index: _,
                    proximal: _,
                    principal,
                    entity,
                    session_id,
                } => policy_sdk::proto::GraphUpdate::EdgeCreated {
                    source,
                    destination,
                    session_id: if session_id.is_empty() {
                        None
                    } else {
                        Some(session_id)
                    },
                    principal,
                    entity,
                },
                sasy_policy::engine::GraphUpdate::EdgeDeleted {
                    source,
                    destination,
                } => policy_sdk::proto::GraphUpdate::EdgeDeleted {
                    source,
                    destination,
                },
                sasy_policy::engine::GraphUpdate::DropSession(_) => {
                    // Plugin SDK doesn't yet expose session lifecycle;
                    // map to a no-op NodeDeleted that the plugin drops.
                    policy_sdk::proto::GraphUpdate::NodeDeleted(String::new())
                }
            })
            .collect();
        self.plugin.apply_graph_updates(sdk_updates)
    }

    fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: &[policy_engine::Action],
        entity: Option<&str>,
        roles: &[String],
        scope: &sasy_common::SessionScope,
        principal: Option<&str>,
        _policy_id: Option<&str>,
    ) -> Result<AuthorizationResponse> {
        // Plugin SDK gets tenant + principal so multi-tenant
        // plugins (and `HasPrincipal` / `HasRole` rules) get
        // the same auth-derived view the in-process engines
        // see. `policy_id` (multi-policy variant binding) still
        // isn't on the plugin wire — variant policies are wired
        // through the gRPC SetPolicy path only.
        let session_id = if scope.is_global() {
            None
        } else {
            Some(scope.session())
        };
        self.plugin.check_authorization(
            current_node_ids,
            actions,
            entity,
            roles,
            session_id,
            Some(scope.tenant()),
            principal,
        )
    }

    fn reset(&self) -> Result<()> {
        self.plugin.reset()
    }

    fn load_rule_metadata(&self, path: &std::path::Path) -> Result<()> {
        self.plugin.load_rule_metadata(path)
    }

    fn get_sync_status(&self) -> sasy_policy::engine::SyncStatus {
        match self.plugin.get_sync_status() {
            Ok(s) => sasy_policy::engine::SyncStatus {
                current_sequence: s.current_sequence,
                node_count: s.node_count as usize,
                edge_count: s.edge_count as usize,
                connected: s.connected,
            },
            Err(_) => sasy_policy::engine::SyncStatus {
                current_sequence: 0,
                node_count: 0,
                edge_count: 0,
                connected: false,
            },
        }
    }

    fn set_connected(&self, connected: bool) {
        self.plugin.set_connected(connected);
    }

    fn set_sequence(&self, seq: i64) {
        self.plugin.set_sequence(seq);
    }
}

// ── PolicyAdapter ────────────────────────────────────
// Bridges sasy_policy::Engine → sasy_refmon::PolicyChecker

struct PolicyAdapter<E: Engine> {
    engine: Arc<E>,
    graph_store: Arc<GraphStore>,
}

impl<E: Engine> sasy_refmon::PolicyChecker for PolicyAdapter<E> {
    async fn check_authorization(
        &self,
        current_node_ids: &[String],
        actions: Vec<policy_engine::Action>,
        entity: Option<&str>,
        roles: &[String],
        scope: sasy_common::SessionScope,
        principal: Option<&str>,
        policy_id: Option<String>,
    ) -> Result<AuthorizationResponse, sasy_refmon::RefmonError> {
        // Session-ownership gate. In the single-binary deployment refmon
        // reaches the engine through this adapter, bypassing the gRPC
        // handler's gate (`PolicyService::check_authorization` →
        // `peek_session_ownership_or_deny`). Without re-checking here, a
        // same-tenant `reference-monitor-user` could evaluate over another
        // principal's session graph (and read its denial traces) just by
        // supplying that session id. Replicate the read-only check: the
        // global ("") session is tenant-shared and unowned (exempt), the
        // owner matches itself, and admin bypasses.
        if !scope.is_global() {
            let owner = self
                .graph_store
                .get_session_owner(&scope)
                .map_err(|e| sasy_refmon::RefmonError::Internal(format!("ownership check: {e}")))?;
            if let Some(owner) = owner {
                let is_owner = principal == Some(owner.as_str());
                let is_admin = roles.iter().any(|r| r == sasy_common::roles::ADMIN);
                if !is_owner && !is_admin {
                    return Err(sasy_refmon::RefmonError::PolicyCheck(format!(
                        "session is owned by another principal: {owner}"
                    )));
                }
            }
        }

        // Refmon and the policy engine are colocated in this binary,
        // so we forward refmon's already-extracted auth context
        // straight into the engine: `entity` is the user-supplied
        // wire actor (preserved verbatim, surfaced as `Entity(id)`),
        // `principal` is the auth-derived identity that the engine
        // stamps as `Principal(id)` and the basis for `HasPrincipal()`
        // / `HasRole(...)`. policy_id is forwarded through.
        self.engine
            .check_authorization(
                current_node_ids,
                &actions,
                entity,
                roles,
                &scope,
                principal,
                policy_id.as_deref(),
            )
            .map_err(|e| sasy_refmon::RefmonError::PolicyCheck(e.to_string()))
    }
}

// ── CLI ──────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "sasy",
    version,
    about = "SASY policy engine: records an agent's message dependency graph \
             and authorizes each tool call against a Datalog policy"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate private, per-launch loopback mutual-TLS credentials for a
    /// co-located client.
    #[cfg(feature = "mtls")]
    GuardTls {
        /// New directory under an existing private parent; never overwritten.
        #[arg(long)]
        output_dir: std::path::PathBuf,
        /// Client identity: 1–64 ASCII letters/digits and . _ @ -; starts alphanumeric.
        #[arg(long)]
        entity: String,
    },
    /// Create local keys, config and TLS in DIR on first use, reuse them
    /// afterwards, and print the client settings as JSON. The Docker image
    /// runs this on start when no configuration is mounted.
    #[cfg(feature = "mtls")]
    LocalInit {
        /// State directory, created if absent.
        dir: std::path::PathBuf,
        /// Client identity on first initialization (default: client).
        #[arg(long)]
        client_entity: Option<String>,
        /// Admin identity on first initialization (default: admin).
        #[arg(long)]
        admin_entity: Option<String>,
        /// Tenant on first initialization (default: default).
        #[arg(long)]
        tenant: Option<String>,
        /// Authentication trust domain on first initialization (default: sasy.local).
        #[arg(long)]
        trust_domain: Option<String>,
    },
    /// Start all gRPC services
    Serve(Box<ServeArgs>),
    /// Initialize credential database and optionally
    /// populate from an env file
    #[cfg(feature = "proxy")]
    InitCredentials {
        /// Credential database path. The default is the file `serve` uses when
        /// started with its default --data-dir; pass the same path to both
        /// commands when either is changed.
        #[arg(long, default_value = "data/graph/credentials.db")]
        credentials_db: String,

        /// File of `NAME=value` lines to populate the database from; blank
        /// lines and lines starting with `#` are ignored.
        #[arg(long)]
        env_file: Option<String>,
    },
    /// Write the Soufflé compile-chain assets this binary carries into a
    /// directory, for operators who want to inspect or patch the chain.
    /// Point `SASY_SOUFFLE_ASSETS` at the result to compile with it.
    #[cfg(feature = "compiler")]
    InstallAssets {
        /// Directory to write the assets into (created if absent).
        dir: String,
    },
}

/// The credential backends `serve` can be pointed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum CredentialBackend {
    /// The local SQLite file. The default, and what local development uses.
    Sqlite,
    /// Process memory, seeded from environment variables at startup.
    Memory,
    /// An OpenBao KV v2 secret engine, read-only.
    Openbao,
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Policy source file (.dl). With `--evaluator souffle` it is compiled at
    /// startup and becomes the tenant default; without it the server boots
    /// deny-all and waits for SetPolicy. With `--policy-plugin` or
    /// `souffle-interpreted` it is read only for the `@deny_message` and
    /// `@suggestion` annotations. Restricted builds ignore it and serve only
    /// their precompiled policies.
    #[arg(long = "policy-metadata", alias = "policy")]
    policy_metadata: Option<String>,

    /// Path to a policy plugin shared library (.so/.dylib).
    #[arg(long = "policy-plugin")]
    policy_plugin: Option<String>,

    /// Policy backend. `souffle` (default): compile each policy to a native
    /// evaluator (needs souffle, g++ and python3). `souffle-interpreted`: run
    /// an external interpreter shim (needs --souffle-policy). `flowlog`: an
    /// external evaluator that is not part of this repository (needs
    /// --flowlog-bin). `stub`: ALLOWS EVERY ACTION, for tests only. Ignored
    /// when --policy-plugin is given.
    #[arg(long, default_value = "souffle")]
    evaluator: String,

    /// Path to an external FlowLog evaluator speaking this binary's evaluator
    /// IPC protocol. Not part of this repository.
    #[arg(long)]
    flowlog_bin: Option<String>,

    /// Program name passed to the bootstrap evaluator. Policies compiled by
    /// this binary are always named `policy_program`; change this only for a
    /// hand-built evaluator.
    #[arg(long, default_value = "policy_program")]
    souffle_program: String,

    /// Path to the Soufflé interpreted shim binary
    #[arg(long, default_value = "souffle-interpreted")]
    souffle_interpreted_bin: String,

    /// Path to the Soufflé policy file (.dl, for interpreted mode).
    #[arg(long)]
    souffle_policy: Option<String>,

    /// Path to a prebuilt Soufflé functor shared library. Used by the
    /// `souffle-interpreted` backend only; this binary does not build it.
    #[arg(long)]
    souffle_functor_lib: Option<String>,

    /// Authentication provider config (JSON): API key, JWT/JWKS, mTLS, or a
    /// chain of them. See config/auth/*.example.json. Required: without it the
    /// server refuses to start, unless SASY_ALLOW_NO_AUTH=1 selects
    /// unauthenticated loopback-only development mode.
    #[arg(long)]
    auth_provider: Option<String>,

    /// YAML mapping each authenticated entity to its roles and tenant (see
    /// config/auth_config.example.yaml). The API key and mTLS providers read
    /// roles only from this file, so without it their entities have no roles.
    /// JWT instead takes roles from the token claim it is configured to read,
    /// unless it is set to use this file; unauthenticated development mode
    /// grants its own local-client roles to entities this file does not list.
    #[arg(long)]
    auth_config: Option<String>,

    /// JSON credential-injection rules: each transform id a policy emits
    /// through `ApplyTransform` names the header or query parameter to fill
    /// and the stored credential to fill it with (see
    /// config/transforms.example.json). Used by the HTTP proxy paths only.
    #[arg(long)]
    transforms: Option<String>,

    /// gRPC listen address
    #[arg(long, default_value = "127.0.0.1:10089")]
    addr: String,

    /// HTTP forward proxy port (0 to disable).
    #[arg(long, default_value = "0")]
    proxy_port: u16,

    /// Address the forward proxy listens on. Loopback by default: the proxy
    /// authenticates nobody and attaches the tenant's stored credentials to
    /// requests the policy allows, so anything that can reach it can spend
    /// those credentials against every host the policy permits. Binding it
    /// off-host is a deliberate act.
    #[arg(long, default_value = "127.0.0.1")]
    proxy_bind: String,

    /// Tenant the forward proxy listener routes traffic into. The
    /// proxy is auth-less per request (raw HTTP/CONNECT), so the
    /// tenant has to be a deployment property of the listener.
    /// Multi-tenant operators run one proxy per tenant and pin
    /// each one with this flag.
    #[arg(long, default_value = "default")]
    proxy_tenant: String,

    /// Let ANY authenticated caller act for another identity by sending
    /// `x-entity`/`x-roles`, without holding the `service-proxy` role. Off by
    /// default. Enable only when every client that can reach --addr is a
    /// trusted relay; otherwise any client can claim `admin`.
    #[arg(long)]
    trust_proxy_headers: bool,

    /// TLS server certificate path (enables TLS)
    #[arg(long)]
    tls_cert: Option<String>,

    /// TLS server private key path
    #[arg(long)]
    tls_key: Option<String>,

    /// TLS CA certificate path (enables mTLS client verification)
    #[arg(long)]
    tls_ca: Option<String>,

    /// Directory for all engine state: the RocksDB graph store, persisted
    /// policies and session bindings, the Soufflé build cache, and by default
    /// credentials.db. Created if missing. A relative path is resolved once,
    /// against the working directory the process started in.
    #[arg(long, default_value = "data/graph")]
    data_dir: String,

    /// Where credentials come from: `sqlite` (the file named by
    /// --credentials-db), `memory` (seeded from the environment, nothing on
    /// disk), or `openbao` (read-only lookups against an OpenBao KV v2
    /// secret engine).
    ///
    /// `memory` is single-tenant: everything seeded from the environment is
    /// stored under one tenant and resolved for every tenant, so on a server
    /// serving more than one tenant it hands each of them the same
    /// credentials. `sqlite` and `openbao` keep tenants apart.
    #[arg(
        long,
        value_enum,
        default_value = "sqlite",
        env = "SASY_CREDENTIAL_BACKEND"
    )]
    credential_backend: CredentialBackend,

    /// Credential DB path (sqlite backend). No default: left unset, the sqlite
    /// backend uses `<data-dir>/credentials.db`, so a server started in a
    /// directory it cannot write to still has a place for the file. The
    /// `memory` and `openbao` backends read no file and create no directory,
    /// whether or not this is given. A relative value is resolved against the
    /// working directory the process started in, once, at startup.
    #[arg(long)]
    credentials_db: Option<String>,

    /// Extra `.env`-style file to seed the memory backend from, on top of the
    /// process environment. Entries here win over the environment.
    ///
    /// Read only by `--credential-backend memory`. With any other backend it
    /// is ignored, and the server says so in a warning at startup — the
    /// setting is also readable from the environment, where a deployment that
    /// never chose the memory backend can inherit it.
    #[arg(long, env = "SASY_CREDENTIALS_ENV_FILE")]
    credentials_env_file: Option<String>,

    /// OpenBao base address, e.g. `https://bao.internal:8200`. Required with
    /// `--credential-backend openbao`.
    // The defaults below repeat sasy-credential's constants as literals
    // because these arguments are parsed even in builds that do not compile
    // the credential stack in.
    #[arg(long, env = "OPENBAO_ADDR")]
    openbao_addr: Option<String>,

    /// KV v2 mount to read from.
    #[arg(long, env = "OPENBAO_MOUNT", default_value = "secret")]
    openbao_mount: String,

    /// Path segments prepended to every lookup.
    #[arg(long, env = "OPENBAO_PATH_PREFIX", default_value = "sasy")]
    openbao_path_prefix: String,

    /// OpenBao namespace, sent as `X-Vault-Namespace`.
    #[arg(long, env = "OPENBAO_NAMESPACE")]
    openbao_namespace: Option<String>,

    /// File holding an OpenBao token. Use this or the AppRole pair, not both.
    #[arg(long, env = "OPENBAO_TOKEN_FILE")]
    openbao_token_file: Option<String>,

    /// AppRole role id (an identifier, not a secret).
    #[arg(long, env = "OPENBAO_APPROLE_ROLE_ID")]
    openbao_approle_role_id: Option<String>,

    /// File holding the AppRole secret id.
    #[arg(long, env = "OPENBAO_APPROLE_SECRET_ID_FILE")]
    openbao_approle_secret_id_file: Option<String>,

    /// PEM file for the certificate authority that signed the OpenBao
    /// server's certificate, for a private CA.
    #[arg(long, env = "OPENBAO_CA_CERT")]
    openbao_ca_cert: Option<String>,

    /// How long a credential read from OpenBao may be reused, in seconds.
    #[arg(long, env = "OPENBAO_CACHE_TTL_SECS", default_value = "60")]
    openbao_cache_ttl_secs: u64,

    /// Per-request timeout for OpenBao calls, in seconds.
    #[arg(long, env = "OPENBAO_TIMEOUT_SECS", default_value = "5")]
    openbao_timeout_secs: u64,

    /// Wall-clock budget for one policy evaluation, in seconds. A check that
    /// overruns it is DENIED (the engine fails closed) while the evaluator
    /// keeps running, so a policy that is slow on some inputs costs exactly
    /// the checks that touch them.
    ///
    /// Must be at least 1. There is no way to switch the budget off: `0` here
    /// does not mean "no timeout" the way it does for --proxy-port — every
    /// evaluation would overrun it and every check would be denied — so the
    /// server refuses to start on it.
    #[arg(long, env = "SASY_QUERY_TIMEOUT_SECS", default_value = "10")]
    query_timeout_secs: u64,

    /// How long an evaluator may owe a reply before it is presumed unable to
    /// answer anything, in seconds. Past this its process is killed and a
    /// fresh one re-bootstraps from the graph store. Must be at least
    /// --query-timeout-secs; the server refuses to start otherwise.
    #[arg(long, env = "SASY_EVALUATOR_STALL_SECS", default_value = "60")]
    evaluator_stall_secs: u64,

    /// Scrub the two strings the LLM oracle sends to the model provider —
    /// `on` (the default) or `off`. The policy's `@llm_check_fn(prompt,
    /// context)` hands both to an outside service, and the context is usually
    /// a message's contents, so on `on` every value the redactor recognizes is
    /// replaced by a marker first. Read from SASY_ORACLE_REDACTION when the
    /// flag is absent; a value neither word leaves it on.
    #[arg(long = "oracle-redaction", value_name = "on|off")]
    oracle_redaction: Option<String>,

    /// Neo4j URI for background graph sync (e.g. bolt://localhost:7687).
    #[arg(long)]
    neo4j_uri: Option<String>,

    /// Neo4j username
    #[arg(long, default_value = "neo4j")]
    neo4j_username: String,

    /// Neo4j password
    #[arg(long, default_value = "observability")]
    neo4j_password: String,

    /// Neo4j database name
    #[arg(long, default_value = "neo4j")]
    neo4j_database: String,

    /// Let non-admin callers upload custom C++ functor source on SetPolicy.
    /// Off unless this flag or SASY_ALLOW_USER_FUNCTORS (below) is set:
    /// functor source is compiled and dynamically loaded as native code
    /// inside the policy engine's own process.
    ///
    /// `--allow-user-functors` (no value) and `--allow-user-functors=sandboxed`
    /// admit a non-admin only on a host where the bubblewrap sandbox works, so
    /// the C++ runs confined. `--allow-user-functors=unsandboxed` admits one on
    /// any host, including a host with no sandbox, where the C++ runs as the
    /// user running this binary — a development setting.
    ///
    /// Also settable as SASY_ALLOW_USER_FUNCTORS: `1`, `true`, `yes`, `on` and
    /// `sandboxed` all mean sandboxed; `unsandboxed` means unsandboxed. The
    /// flag wins when both are given.
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "sandboxed",
        value_parser = ["sandboxed", "unsandboxed"],
    )]
    allow_user_functors: Option<String>,
}

/// Read the functor opt-in the operator may have set in the environment
/// instead of on the command line.
///
/// `1`, `true`, `yes`, `on` and `sandboxed` select
/// [`UserFunctors::Sandboxed`]; `unsandboxed` selects
/// [`UserFunctors::Unsandboxed`]; unset, empty, `0`, `false`, `no` and `off`
/// select [`UserFunctors::Refuse`]. Anything else is an error: a value nobody
/// can interpret must not quietly become one of the three answers, least of
/// all by resembling one.
///
/// Takes the value rather than the variable's name so the rule is testable
/// without mutating the process environment.
fn env_user_functors(value: Option<&str>) -> Result<UserFunctors, String> {
    match value
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "" | "0" | "false" | "no" | "off" => Ok(UserFunctors::Refuse),
        "1" | "true" | "yes" | "on" | "sandboxed" => Ok(UserFunctors::Sandboxed),
        "unsandboxed" => Ok(UserFunctors::Unsandboxed),
        other => Err(format!(
            "SASY_ALLOW_USER_FUNCTORS={other:?} is not a value this binary understands. \
             Use 'sandboxed' (also spelled 1, true, yes, on), 'unsandboxed', or leave it \
             unset."
        )),
    }
}

/// Build the policy service's configuration from the operator's two inputs for
/// the functor gate: the `--allow-user-functors` flag (already validated by
/// clap to be absent, `sandboxed` or `unsandboxed`) and the raw value of
/// `SASY_ALLOW_USER_FUNCTORS`. The flag wins when both are given.
///
/// The environment value is interpreted whenever it is set, flag or no flag,
/// and only then does the flag override the answer. So a value nobody can
/// read is a boot error even when the flag would have won: an operator who
/// set the variable meant something by it, and the binary must not run on
/// while that meaning is unknown.
///
/// `sandbox_available: None` means "ask the host when the question is asked",
/// which is what the gate does; see the call site for why a startup snapshot
/// would be wrong.
fn policy_service_config(
    allow_user_functors_flag: Option<&str>,
    allow_user_functors_env: Option<&str>,
) -> Result<sasy_policy::service::PolicyServiceConfig, String> {
    // Read the environment first, and fail on a value nobody can interpret,
    // before the flag gets its chance to override the answer.
    let from_env = env_user_functors(allow_user_functors_env)?;
    let user_functors = match allow_user_functors_flag {
        Some("sandboxed") => UserFunctors::Sandboxed,
        Some("unsandboxed") => UserFunctors::Unsandboxed,
        Some(other) => {
            return Err(format!(
                "--allow-user-functors={other:?} is not a value this binary understands; \
                 use 'sandboxed' or 'unsandboxed'"
            ))
        }
        None => from_env,
    };
    Ok(sasy_policy::service::PolicyServiceConfig {
        user_functors,
        sandbox_available: None,
    })
}

/// Resolve the oracle-redaction switch from the operator's two inputs: the
/// `--oracle-redaction` flag and the raw value of `SASY_ORACLE_REDACTION`.
/// The flag wins when both are given.
///
/// Absent both, the answer is on. A value that is neither word is not an
/// answer, so it is reported and the switch stays on: the failure of an
/// operator's input is not a reason to start sending recorded content to an
/// outside service in the clear. That is also why an uninterpretable value
/// does not stop the boot the way the functor gate's does — there the safe
/// reading is to refuse, here it is to redact.
///
/// The environment is read first, so a value nobody can interpret is reported
/// even when the flag goes on to override it.
fn oracle_redaction_setting(flag: Option<&str>, env: Option<&str>) -> bool {
    let from_env = read_on_off(env, "SASY_ORACLE_REDACTION");
    match flag {
        Some(value) => read_on_off(Some(value), "--oracle-redaction"),
        None => from_env,
    }
}

/// `on` / `off`, spelled in any case and with any surrounding space. Anything
/// else is warned about, under the name the operator wrote it as, and read as
/// on.
fn read_on_off(value: Option<&str>, source: &str) -> bool {
    match value {
        None => true,
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "on" => true,
            "off" => false,
            _ => {
                warn!(
                    "{source}={raw:?} is not a value this binary understands; use 'on' or \
                     'off'. Leaving oracle redaction on."
                );
                true
            }
        },
    }
}

/// The one warning the unsandboxed opt-in owes the operator at startup: on
/// this host the confinement the sandboxed setting relies on does not exist,
/// so a non-admin's C++ will be compiled and run with nothing around it.
///
/// `None` in every other combination — with the sandbox present the
/// unsandboxed setting changes nothing about how the code runs, and the other
/// two settings never admit a non-admin unconfined.
fn unconfined_functor_warning(
    user_functors: UserFunctors,
    sandbox_available: bool,
) -> Option<&'static str> {
    match (user_functors, sandbox_available) {
        (UserFunctors::Unsandboxed, false) => Some(
            "--allow-user-functors=unsandboxed: non-admin functor source will compile \
             and run unconfined on this host (no working bubblewrap sandbox), as the \
             user running sasy serve",
        ),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install TLS crypto provider before any TLS operations
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        #[cfg(feature = "mtls")]
        Command::GuardTls { output_dir, entity } => guard_tls::generate(&output_dir, &entity),
        #[cfg(feature = "mtls")]
        Command::LocalInit {
            dir,
            client_entity,
            admin_entity,
            tenant,
            trust_domain,
        } => local_init::run(
            &dir,
            &local_init::IdentityOptions {
                client_entity,
                admin_entity,
                tenant,
                trust_domain,
            },
        ),
        Command::Serve(args) => serve(*args).await,
        #[cfg(feature = "proxy")]
        Command::InitCredentials {
            credentials_db,
            env_file,
        } => init_credentials(credentials_db, env_file),
        #[cfg(feature = "compiler")]
        Command::InstallAssets { dir } => install_assets(&dir),
    }
}

/// Write the embedded compile-chain assets into `dir` and say what landed.
///
/// The point is an on-disk copy an operator can read, diff or patch; the
/// resulting directory is what `SASY_SOUFFLE_ASSETS` expects. A directory that
/// cannot be written is an error, not a warning — a half-installed chain would
/// fail later, at a policy compile, with a worse message.
#[cfg(feature = "compiler")]
fn install_assets(dir: &str) -> Result<()> {
    let dir = std::path::Path::new(dir);
    sasy_policy::assets::materialize(dir, sasy_policy::assets::ASSETS)
        .with_context(|| format!("failed to write Soufflé assets to {}", dir.display()))?;
    println!(
        "Wrote {} Soufflé compile-chain assets (set {}) to {}:",
        sasy_policy::assets::ASSETS.len(),
        sasy_policy::assets::asset_set_hash(),
        dir.display()
    );
    for (name, _) in sasy_policy::assets::ASSETS {
        println!("  {name}");
    }
    println!("Compile with them: SASY_SOUFFLE_ASSETS={}", dir.display());
    Ok(())
}

async fn serve(args: ServeArgs) -> Result<()> {
    let ServeArgs {
        policy_metadata: policy_file,
        policy_plugin: policy_plugin_path,
        evaluator: evaluator_backend,
        souffle_program,
        souffle_interpreted_bin,
        souffle_policy,
        souffle_functor_lib,
        flowlog_bin,
        auth_provider: auth_provider_file,
        auth_config: auth_config_file,
        transforms: transforms_file,
        addr,
        proxy_port,
        proxy_bind,
        proxy_tenant,
        trust_proxy_headers,
        tls_cert,
        tls_key,
        tls_ca,
        data_dir,
        credential_backend,
        credentials_db,
        credentials_env_file,
        openbao_addr,
        openbao_mount,
        openbao_path_prefix,
        openbao_namespace,
        openbao_token_file,
        openbao_approle_role_id,
        openbao_approle_secret_id_file,
        openbao_ca_cert,
        openbao_cache_ttl_secs,
        openbao_timeout_secs,
        query_timeout_secs,
        evaluator_stall_secs,
        oracle_redaction,
        neo4j_uri,
        neo4j_username,
        neo4j_password,
        neo4j_database,
        allow_user_functors,
    } = args;

    // Published before anything can start an evaluator, so every backend's
    // oracle callbacks read the operator's setting and none reads the default
    // by accident.
    sasy_policy::oracle_redaction::set_oracle_redaction(oracle_redaction_setting(
        oracle_redaction.as_deref(),
        std::env::var("SASY_ORACLE_REDACTION").ok().as_deref(),
    ));

    // Refuse the impossible pair at startup, where an operator sees it, rather
    // than as a surprise kill on the first slow query.
    let deadlines = sasy_policy::session_evaluator::EvaluationDeadlines::new(
        std::time::Duration::from_secs(query_timeout_secs),
        std::time::Duration::from_secs(evaluator_stall_secs),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    // The flag and the env var are both operator input; the flag wins. An
    // uninterpretable env value stops the boot rather than resolving to one of
    // the three answers by accident.
    let policy_service_config = policy_service_config(
        allow_user_functors.as_deref(),
        std::env::var("SASY_ALLOW_USER_FUNCTORS").ok().as_deref(),
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    let user_functors = policy_service_config.user_functors;
    // Probed here for the startup warnings only; the gate does not read this
    // value (`PolicyServiceConfig::sandbox_available: None` means it calls
    // `sandbox_available()` itself). What that call re-reads on every upload is
    // the operator's switches — `DISABLE_BWRAP`, `SASY_EVALUATOR_BWRAP` — and
    // whether a `bwrap` is on PATH at all. What it does NOT re-read is whether
    // that bwrap can create namespaces: the probe behind that answer runs once
    // per process, at the first call, and its verdict is cached for the life of
    // the process. So a `bwrap` upgraded, downgraded or otherwise replaced
    // under a running binary keeps the old verdict until the binary restarts.
    let sandbox_available = sasy_policy::sandbox::sandbox_available();
    // Whether the evaluator's syscall deny-list is in place before its first
    // instruction, which is a narrower question than whether it is confined at
    // all: on a bubblewrap older than 0.4.0 the namespaces are there from the
    // start and the deny-list is not, so a functor's static constructors run
    // inside the sandbox with the full syscall surface. Reported next to
    // `sandbox_available` so neither line reads as a promise about the other.
    let seccomp_pre_exec = sasy_policy::sandbox::seccomp_pre_exec_available();
    sasy_policy::sandbox::log_seccomp_pre_exec_status();
    if user_functors != UserFunctors::Refuse {
        tracing::warn!(
            sandbox_available,
            seccomp_pre_exec,
            opt_in = user_functors.as_flag_value(),
            "--allow-user-functors: callers without the admin role may upload C++ \
             functor source, which is compiled and loaded into the policy engine's \
             process. Persisted source admitted this way is re-checked against these \
             same settings every time it is loaded."
        );
    }
    if let Some(warning) = unconfined_functor_warning(user_functors, sandbox_available) {
        tracing::warn!("{warning}");
    }

    // Create data directories
    std::fs::create_dir_all(&data_dir)
        .with_context(|| format!("failed to create graph data dir {data_dir}"))?;
    // Resolve --data-dir once, here, and use the absolute form from now on.
    // The default (`data/graph`) is relative to the directory the server was
    // started in; everything derived from it — the graph store, the Soufflé
    // build cache, the materialized compile-chain assets, the bootstrap build
    // directory — must keep naming the same place for the life of the process,
    // whatever any later code does with the working directory.
    let data_dir: String = std::fs::canonicalize(&data_dir)
        .with_context(|| format!("failed to resolve data dir {data_dir}"))?
        .to_string_lossy()
        .into_owned();
    // Where the credential file lives, decided once here and absolute from now
    // on. Unset, it hangs off the (already absolute) data directory, so a
    // server started in a directory it cannot write to still boots.
    let credentials_db_set = credentials_db.is_some();
    let startup_cwd = std::env::current_dir()
        .context("failed to read the working directory the server started in")?;
    let credentials_db = resolve_credentials_db(credentials_db.as_deref(), &data_dir, &startup_cwd)
        .to_string_lossy()
        .into_owned();
    #[cfg(feature = "proxy")]
    ensure_credentials_dir(credential_backend, std::path::Path::new(&credentials_db))?;

    // Soufflé build cache — compile path only. The restricted build has no
    // compiler, so the whole cache machinery (and its g++/souffle version
    // probes) is gated out.
    #[cfg(feature = "compiler")]
    {
        // Default the cache to a subdir of --data-dir so it survives restarts
        // on a persistent volume. An explicit env var (set/empty/"off")
        // takes precedence so tests and CI runs can opt out.
        if std::env::var_os("SASY_SOUFFLE_BUILD_CACHE_DIR").is_none() {
            let cache_dir = std::path::Path::new(&data_dir).join("souffle-build-cache");
            std::env::set_var("SASY_SOUFFLE_BUILD_CACHE_DIR", &cache_dir);
            info!(
                cache_dir = %cache_dir.display(),
                "Soufflé build cache enabled (default to <data-dir>/souffle-build-cache)"
            );
        }
        // Prune stale build-cache entries before any compile uses it: clears
        // everything on a toolchain/key-schema change (those entries are
        // unreachable), and age-prunes long-idle binaries. No-op if disabled.
        sasy_policy::souffle_cache::prune_at_startup();
    }

    // ── Auth config ──────────────────────────────
    let auth_config = if let Some(ref path) = auth_config_file {
        let cfg = AuthConfig::load(path).context("failed to load auth config")?;
        info!(path, entities = cfg.entities.len(), "loaded auth config");
        Some(cfg)
    } else {
        None
    };

    // ── Auth interceptor ─────────────────────────
    let auth_provider: Arc<dyn sasy_auth::AuthProvider> = if let Some(ref path) = auth_provider_file
    {
        let p = sasy_auth::load_provider_from_file(std::path::Path::new(path), auth_config.clone())
            .context("failed to load auth provider")?;
        info!(path, provider = p.name(), "loaded auth provider");
        p
    } else {
        // No --auth-provider → PassthroughAuthProvider, which always succeeds
        // and trusts the caller-supplied `x-entity` header. That means ANY
        // caller can assert any identity (including `admin`) with no
        // credential. Fail closed: refuse to start unless the operator
        // explicitly opts into no-auth mode for local development.
        let allow_no_auth = sasy_common::env_flag("SASY_ALLOW_NO_AUTH");
        if !allow_no_auth {
            anyhow::bail!(
                "no --auth-provider configured: refusing to start in passthrough \
                 (no-auth) mode, where any caller can assert admin via the \
                 `x-entity` header. Pass --auth-provider <file> (the provider \
                 examples are under config/auth/), or set SASY_ALLOW_NO_AUTH=1 \
                 for local dev."
            );
        }
        warn!(
            "SASY_ALLOW_NO_AUTH set and no --auth-provider configured: running in \
             PASSTHROUGH mode — any loopback caller is granted the local-client \
             role set (no auth_config/entity/certs needed). Trust boundary is \
             loopback + the local OS user. LOCAL SINGLE-USER ONLY; never expose."
        );
        // Local-trust default: grant the roles a co-located client needs, so a
        // local deployment works with no auth_config or entity setup.
        // (auth_config, when present, still wins for a recognized entity.)
        let default_roles = vec![
            sasy_common::roles::REFERENCE_MONITOR_USER.to_string(),
            sasy_common::roles::OBSERVABILITY_WRITER.to_string(),
            sasy_common::roles::OBSERVABILITY_READER.to_string(),
            sasy_common::roles::CREDENTIAL_READER.to_string(),
            "policy-client".to_string(),
        ];
        Arc::new(PassthroughAuthProvider::with_default_roles(
            auth_config.clone(),
            default_roles,
        ))
    };
    let auth_interceptor = AuthInterceptor::new(auth_provider);

    // ── Shared state ─────────────────────────────
    let graph = Arc::new(GraphStore::new(Some(&data_dir)).context("failed to open graph store")?);

    // ── Neo4j background sync (optional; gated out of the restricted build) ──
    #[cfg(feature = "neo4j-sync")]
    if let Some(ref uri) = neo4j_uri {
        // The database name is the one value the mirror puts into a request
        // path rather than a Cypher parameter, so it is checked against
        // Neo4j's name grammar before anything is sent.
        sasy_graph::neo4j_sync::check_database_name(&neo4j_database).map_err(|why| {
            anyhow::anyhow!(
                "--neo4j-database {neo4j_database:?} is not a valid Neo4j database name: {why}"
            )
        })?;
        let config = sasy_graph::neo4j_sync::Neo4jSyncConfig {
            uri: uri.clone(),
            username: neo4j_username.clone(),
            password: neo4j_password.clone(),
            database: neo4j_database.clone(),
        };
        sasy_graph::neo4j_sync::start(Arc::clone(&graph), config);
        info!(uri, "Neo4j background sync enabled");
    }

    // ── TLS config ────────────────────────────────
    let server_tls = if tls_cert.is_some() || tls_key.is_some() {
        let tls = sasy_auth::TlsConfig {
            cert_path: tls_cert,
            key_path: tls_key,
            ca_path: tls_ca,
        };
        if tls.is_mtls() {
            info!("mTLS enabled (client certificates required)");
        } else {
            info!("TLS enabled");
        }
        Some(tls)
    } else {
        None
    };

    // `mut` so the restricted branch can pin the policy lock to the baked
    // default's content hash (unused in the full build, where it stays `None`).
    #[cfg_attr(not(feature = "restricted"), allow(unused_mut))]
    let mut svc_config = ServiceConfig {
        auth_interceptor,
        auth_config,
        transforms_file,
        addr,
        proxy_port,
        proxy_bind,
        proxy_tenant,
        trust_proxy_headers,
        tls_config: server_tls,
        credentials: CredentialConfig {
            backend: credential_backend,
            db_path: credentials_db,
            db_path_set: credentials_db_set,
            env_file: credentials_env_file,
            openbao_addr,
            openbao_mount,
            openbao_path_prefix,
            openbao_namespace,
            openbao_token_file,
            openbao_approle_role_id,
            openbao_approle_secret_id_file,
            openbao_ca_cert,
            openbao_cache_ttl_secs,
            openbao_timeout_secs,
        },
        locked_policy_hashes: None,
        policy_service_config,
    };

    // ── Policy engine ────────────────────────────
    // Restricted release build: serve ONLY the baked precompiled policy pack —
    // no sugar.py/souffle/g++ at runtime. The full selection below is compiled
    // out (and so is the LLM oracle).
    #[cfg(feature = "restricted")]
    let serve_result = {
        let _ = (&policy_plugin_path, &evaluator_backend, &policy_file); // configured-away
        #[cfg(not(feature = "proxy"))]
        warn_ignored_credential_flags(&svc_config.credentials);
        // The config goes in with the engine so the lazy install — which
        // compiles a persisted policy on first dispatch — asks the
        // custom-functor question of the same settings the service's boot
        // replay asks, exactly as the full build does above.
        let (engine, locked_hashes) = restricted::build_engine(
            Arc::clone(&graph),
            &data_dir,
            deadlines,
            svc_config.policy_service_config.clone(),
        )?;
        // Pin SetPolicy to the baked default so a co-located caller can't bind a
        // softer policy (the restricted pack ships no allow-all; the lock is the
        // belt-and-suspenders for any future curated profile). See restricted.rs.
        svc_config.locked_policy_hashes = Some(locked_hashes);
        start_services(engine, graph, svc_config).await
    };

    // Full build: --policy-plugin > --evaluator flag.
    #[cfg(not(feature = "restricted"))]
    let serve_result = if let Some(ref plugin_path) = policy_plugin_path {
        let loader = policy_sdk::PluginLoader::load(std::path::Path::new(plugin_path))
            .context("failed to load policy plugin")?;
        let plugin = PluginEngine::new(std::sync::Arc::new(loader))
            .context("failed to create plugin engine")?;

        if let Some(ref path) = policy_file {
            plugin
                .load_rule_metadata(std::path::Path::new(path))
                .context("failed to load policy metadata")?;
            info!(path, "loaded policy metadata");
        }

        let engine = Arc::new(PluginEngineAdapter { plugin });
        info!(plugin_path, "using plugin policy engine");
        start_services(engine, graph, svc_config.clone()).await
    } else {
        info!(evaluator = %evaluator_backend, "initializing evaluator");
        let Some(be) = sasy_common::Backend::from_wire(&evaluator_backend) else {
            anyhow::bail!("unknown evaluator backend: {}", evaluator_backend);
        };
        match be {
            sasy_common::Backend::Souffle => {
                use sasy_policy::evaluator::manager::EvaluatorProcess;

                // Compile the bootstrap evaluator from --policy, or
                // fall back to the built-in deny-all policy when no
                // --policy is supplied. The build cache hardlinks
                // hits, so re-deriving on every start-up is cheap on
                // the second boot.
                // Under the (absolute) data directory, never the working one.
                let build_dir =
                    sasy_policy::compiler::bootstrap_build_dir(std::path::Path::new(&data_dir))
                        .context("failed to prepare the bootstrap build directory")?;
                let assets = sasy_policy::compiler::SouffleAssets::discover()
                    .context("failed to discover Soufflé build assets")?;
                let eval_bin = if let Some(ref policy_path) = policy_file {
                    info!("Compiling Soufflé evaluator from {}", policy_path);
                    let policy_path = std::path::Path::new(policy_path);
                    // Link the policy's companion functor source (e.g.
                    // `<name>_functors.cpp`) if present, so a bootstrap
                    // `--policy` that declares custom functors links them
                    // the same way a SetPolicy upload would. Without this a
                    // policy that declares custom functors would fail to
                    // link them.
                    let functors = sasy_policy::compiler::find_functors(policy_path);
                    let result = sasy_policy::compiler::compile_souffle_from_file(
                        policy_path,
                        functors.as_deref(),
                        &build_dir,
                        &assets,
                    )
                    .context("failed to compile Soufflé policy")?;
                    info!(
                        "Soufflé evaluator compiled: {}",
                        result.binary_path.display()
                    );
                    result.binary_path
                } else {
                    info!("No --policy supplied; bootstrapping with built-in deny-all policy");
                    let result = sasy_policy::compiler::compile_souffle_with_assets(
                        BOOTSTRAP_DENY_ALL_POLICY,
                        None,
                        &build_dir,
                        &assets,
                    )
                    .context("failed to compile bootstrap deny-all policy")?;
                    info!(
                        "Bootstrap evaluator compiled: {}",
                        result.binary_path.display()
                    );
                    result.binary_path
                };

                // Each session spawns its own evaluator subprocess lazily;
                // per-session evaluators are the unit of concurrency.
                let factory_eval_bin = eval_bin.clone();
                let factory_program = souffle_program.clone();
                let factory: sasy_policy::session_evaluator::EvaluatorFactory =
                    Arc::new(move || {
                        EvaluatorProcess::souffle(factory_eval_bin.clone(), factory_program.clone())
                            .map(|e| Arc::new(e) as Arc<dyn sasy_policy::Evaluator>)
                            .map_err(|e| {
                                sasy_policy::EvaluatorError::InitError(format!(
                                    "Soufflé subprocess: {}",
                                    e
                                ))
                            })
                    });

                // Initialize the LLM service for oracle callbacks during Soufflé evaluation.
                sasy_policy::llm::init();

                start_evaluator_services(
                    factory,
                    "souffle",
                    graph,
                    svc_config.clone(),
                    policy_file.as_deref(),
                    deadlines,
                )
                .await
            }
            sasy_common::Backend::SouffleInterpreted => {
                use sasy_policy::evaluator::manager::{EvaluatorProcess, EvaluatorProcessConfig};
                let policy_path =
                    souffle_policy.context("--souffle-policy required for souffle-interpreted")?;
                let mut args = vec![policy_path, "souffle".into()];
                if let Some(lib) = souffle_functor_lib {
                    args.push(lib);
                }
                let config = EvaluatorProcessConfig {
                    program: souffle_interpreted_bin,
                    args,
                    backend: "souffle-interpreted".into(),
                    env: Vec::new(),
                };
                let factory_config = config.clone();
                let factory: sasy_policy::session_evaluator::EvaluatorFactory =
                    Arc::new(move || {
                        EvaluatorProcess::new(factory_config.clone())
                            .map(|e| Arc::new(e) as Arc<dyn sasy_policy::Evaluator>)
                    });
                start_evaluator_services(
                    factory,
                    "souffle-interpreted",
                    graph,
                    svc_config.clone(),
                    policy_file.as_deref(),
                    deadlines,
                )
                .await
            }
            sasy_common::Backend::Flowlog => {
                use sasy_policy::evaluator::manager::{EvaluatorProcess, EvaluatorProcessConfig};
                let bin = flowlog_bin
                    .clone()
                    .context("--flowlog-bin is required for flowlog backend")?;
                let config = EvaluatorProcessConfig {
                    program: bin,
                    args: vec![],
                    backend: "flowlog".into(),
                    env: Vec::new(),
                };
                let factory_config = config.clone();
                let factory: sasy_policy::session_evaluator::EvaluatorFactory =
                    Arc::new(move || {
                        EvaluatorProcess::new(factory_config.clone())
                            .map(|e| Arc::new(e) as Arc<dyn sasy_policy::Evaluator>)
                    });
                start_evaluator_services(
                    factory,
                    "flowlog",
                    graph,
                    svc_config.clone(),
                    policy_file.as_deref(),
                    deadlines,
                )
                .await
            }
            sasy_common::Backend::Stub => {
                let engine = StubEngine::new();
                info!("using stub engine (allows everything)");
                start_services(engine, graph, svc_config.clone()).await
            }
        }
    };

    serve_result
}

/// Where the sqlite credential database lives.
///
/// `given` is `--credentials-db` as the operator wrote it. There is no clap
/// default: absent, the file hangs off `data_dir` — which `serve` has already
/// resolved to an absolute path — so a server whose working directory is not
/// writable still has somewhere to put it. A relative value is resolved
/// against `startup_cwd`, the working directory the process started in, and
/// the absolute result is what everything downstream sees.
fn resolve_credentials_db(
    given: Option<&str>,
    data_dir: &str,
    startup_cwd: &std::path::Path,
) -> std::path::PathBuf {
    match given {
        Some(path) => startup_cwd.join(path),
        None => std::path::Path::new(data_dir).join("credentials.db"),
    }
}

/// The directory `serve` must create for the credential database, if any.
///
/// Only the sqlite backend keeps a file. `memory` seeds from the environment
/// and `openbao` reads a remote secret engine, so neither may create a
/// directory — on a read-only deployment that would be a boot failure over a
/// file nothing opens.
#[cfg(feature = "proxy")]
fn credentials_dir_to_create(
    backend: CredentialBackend,
    db_path: &std::path::Path,
) -> Option<&std::path::Path> {
    if backend != CredentialBackend::Sqlite {
        return None;
    }
    db_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
}

/// Create the credential database's directory, when the backend keeps one.
///
/// The error names the directory: a boot that dies here on a read-only
/// deployment has to say which path it could not create.
#[cfg(feature = "proxy")]
fn ensure_credentials_dir(backend: CredentialBackend, db_path: &std::path::Path) -> Result<()> {
    if let Some(dir) = credentials_dir_to_create(backend, db_path) {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create credentials dir {}", dir.display()))?;
    }
    Ok(())
}

/// Everything `serve` needs to decide where credentials come from.
#[derive(Clone)]
#[cfg_attr(not(feature = "proxy"), allow(dead_code))]
struct CredentialConfig {
    backend: CredentialBackend,
    /// sqlite backend: the database file, absolute.
    db_path: String,
    /// Whether `--credentials-db` was given a value, as opposed to
    /// [`db_path`](Self::db_path) being derived from the data directory.
    db_path_set: bool,
    /// memory backend: an extra `.env`-style file to seed from.
    env_file: Option<String>,
    openbao_addr: Option<String>,
    openbao_mount: String,
    openbao_path_prefix: String,
    openbao_namespace: Option<String>,
    openbao_token_file: Option<String>,
    openbao_approle_role_id: Option<String>,
    openbao_approle_secret_id_file: Option<String>,
    openbao_ca_cert: Option<String>,
    openbao_cache_ttl_secs: u64,
    openbao_timeout_secs: u64,
}

/// Which credential settings the operator gave a value of their own.
///
/// Compared against the defaults, so a flag left alone is not reported. Used
/// by the restricted build, which parses all of them and builds no credential
/// stack at all.
#[cfg_attr(feature = "proxy", allow(dead_code))]
fn credential_flags_set(config: &CredentialConfig) -> Vec<&'static str> {
    let mut set = Vec::new();
    if config.backend != CredentialBackend::Sqlite {
        set.push("--credential-backend");
    }
    if config.db_path_set {
        set.push("--credentials-db");
    }
    if config.env_file.is_some() {
        set.push("--credentials-env-file");
    }
    if config.openbao_addr.is_some() {
        set.push("--openbao-addr");
    }
    if config.openbao_mount != "secret" {
        set.push("--openbao-mount");
    }
    if config.openbao_path_prefix != "sasy" {
        set.push("--openbao-path-prefix");
    }
    if config.openbao_namespace.is_some() {
        set.push("--openbao-namespace");
    }
    if config.openbao_token_file.is_some() {
        set.push("--openbao-token-file");
    }
    if config.openbao_approle_role_id.is_some() {
        set.push("--openbao-approle-role-id");
    }
    if config.openbao_approle_secret_id_file.is_some() {
        set.push("--openbao-approle-secret-id-file");
    }
    if config.openbao_ca_cert.is_some() {
        set.push("--openbao-ca-cert");
    }
    if config.openbao_cache_ttl_secs != 60 {
        set.push("--openbao-cache-ttl-secs");
    }
    if config.openbao_timeout_secs != 5 {
        set.push("--openbao-timeout-secs");
    }
    set
}

/// Say so when credential settings were given to a build that has no
/// credential stack.
///
/// The restricted build is built without the `proxy` feature, so nothing
/// reads these: clap accepts every one of them, the process starts, and no
/// credential source is ever built. Each of them is also readable from the
/// environment, so they can arrive without anyone typing them.
#[cfg(not(feature = "proxy"))]
fn warn_ignored_credential_flags(config: &CredentialConfig) {
    let set = credential_flags_set(config);
    if !set.is_empty() {
        warn!(
            flags = ?set,
            "this build has no credential stack; these credential settings were \
             accepted and do nothing"
        );
    }
}

/// Whether `--credentials-env-file` was set for a backend that never reads it.
///
/// Only the memory backend seeds from that file. The setting is also readable
/// from the environment, so a deployment that never chose the memory backend
/// can inherit it; silently ignoring it left the operator with a server that
/// denies or under-injects and nothing in the log pointing at the cause.
#[cfg_attr(not(feature = "proxy"), allow(dead_code))]
fn env_file_is_ignored(config: &CredentialConfig) -> bool {
    config.env_file.is_some() && config.backend != CredentialBackend::Memory
}

/// Build the credential source the operator asked for.
///
/// Every backend is established here, at startup, rather than on the first
/// injection: a credential store that cannot be reached is a deployment
/// mistake, and it should stop the process rather than turn into a denial on
/// somebody's first authorized request.
#[cfg(feature = "proxy")]
async fn build_credential_source(config: &CredentialConfig) -> Result<Arc<dyn CredentialSource>> {
    if env_file_is_ignored(config) {
        warn!(
            path = ?config.env_file,
            backend = ?config.backend,
            "--credentials-env-file / SASY_CREDENTIALS_ENV_FILE is set but only the \
             memory backend reads it; nothing was seeded from this file"
        );
    }
    match config.backend {
        CredentialBackend::Sqlite => {
            let source = SqliteSource::open(&config.db_path)?;
            info!(backend = "sqlite", path = %config.db_path, "credential backend ready");
            Ok(Arc::new(source))
        }
        CredentialBackend::Memory => {
            let source = MemorySource::new()?;
            let file_content = match config.env_file {
                Some(ref path) => Some(
                    std::fs::read_to_string(path)
                        .with_context(|| format!("failed to read credentials env file: {path}"))?,
                ),
                None => None,
            };
            let vars = credential_seed_vars(readable_env_vars(), file_content.as_deref());
            // Counts only — a credential in this backend exists nowhere but
            // memory, and a log line would be a copy that outlives it.
            let count = source.seed(vars)?;
            info!(backend = "memory", count, "credential backend ready");
            Ok(Arc::new(source))
        }
        CredentialBackend::Openbao => {
            let addr = config
                .openbao_addr
                .clone()
                .context("--credential-backend openbao requires --openbao-addr")?;
            let auth = match (
                &config.openbao_token_file,
                &config.openbao_approle_role_id,
                &config.openbao_approle_secret_id_file,
            ) {
                (Some(file), None, None) => OpenBaoAuth::TokenFile(file.clone()),
                (None, Some(role_id), Some(secret_id_file)) => OpenBaoAuth::AppRole {
                    role_id: role_id.clone(),
                    secret_id_file: secret_id_file.clone(),
                },
                _ => anyhow::bail!(
                    "--credential-backend openbao needs exactly one way to authenticate: \
                     --openbao-token-file, or --openbao-approle-role-id together with \
                     --openbao-approle-secret-id-file"
                ),
            };
            let bao = OpenBaoConfig {
                addr: addr.clone(),
                mount: config.openbao_mount.clone(),
                path_prefix: config.openbao_path_prefix.clone(),
                namespace: config.openbao_namespace.clone(),
                auth,
                ca_cert: config.openbao_ca_cert.clone(),
                cache_ttl: std::time::Duration::from_secs(config.openbao_cache_ttl_secs),
                timeout: std::time::Duration::from_secs(config.openbao_timeout_secs),
            };
            let source = OpenBaoSource::connect(bao)
                .await
                .context("failed to establish the OpenBao credential backend")?;
            info!(backend = "openbao", addr = %addr, "credential backend ready");
            Ok(Arc::new(source))
        }
    }
}

#[derive(Clone)]
struct ServiceConfig {
    auth_interceptor: AuthInterceptor,
    auth_config: Option<AuthConfig>,
    transforms_file: Option<String>,
    addr: String,
    proxy_port: u16,
    /// See `ServeArgs::proxy_bind`.
    proxy_bind: String,
    proxy_tenant: String,
    trust_proxy_headers: bool,
    tls_config: Option<sasy_auth::TlsConfig>,
    /// Which credential backend to build, and everything it needs. Read only
    /// by the full build's credential source + transform executor; ignored
    /// when the proxy/credential stack is gated out (the restricted build).
    #[cfg_attr(not(feature = "proxy"), allow(dead_code))]
    credentials: CredentialConfig,
    /// Restricted builds only: the set of curated baked policy hashes SetPolicy
    /// may bind. `None` in the full build (policy authoring allowed).
    #[cfg_attr(not(feature = "restricted"), allow(dead_code))]
    locked_policy_hashes: Option<std::collections::HashSet<String>>,
    /// Operator decisions the policy service consults per request — today the
    /// custom-functor gate and what it knows about this host's sandbox.
    policy_service_config: sasy_policy::service::PolicyServiceConfig,
}

/// Wrap a per-session evaluator `factory` in an [`EvaluatorEngine`], load the
/// optional `--policy` rule metadata, and hand off to [`start_services`]. The
/// shared tail of the Souffle / souffle-interpreted / flowlog backend arms.
async fn start_evaluator_services(
    factory: sasy_policy::session_evaluator::EvaluatorFactory,
    backend: &str,
    graph: Arc<GraphStore>,
    svc_config: ServiceConfig,
    policy_file: Option<&str>,
    deadlines: sasy_policy::session_evaluator::EvaluationDeadlines,
) -> Result<()> {
    let engine = Arc::new(
        sasy_policy::EvaluatorEngine::with_deadlines(
            factory,
            backend.to_string(),
            Arc::clone(&graph),
            deadlines,
        )
        // The lazy install compiles persisted policies on first dispatch,
        // functor source included, so it runs the same gate the upload ran
        // — against these settings, in this process.
        .with_policy_service_config(svc_config.policy_service_config.clone()),
    );
    if let Some(path) = policy_file {
        engine
            .load_rule_metadata(std::path::Path::new(path))
            .context("failed to load policy metadata")?;
        info!(path, "loaded policy metadata");
    }
    start_services(engine, graph, svc_config).await
}

async fn start_services<E: Engine + 'static>(
    engine: Arc<E>,
    graph: Arc<GraphStore>,
    config: ServiceConfig,
) -> Result<()> {
    let ServiceConfig {
        auth_interceptor,
        auth_config,
        transforms_file,
        addr,
        proxy_port,
        proxy_bind,
        proxy_tenant,
        trust_proxy_headers,
        tls_config,
        credentials,
        locked_policy_hashes,
        policy_service_config,
    } = config;
    // Consumed only by the restricted policy lock below; in the full build it
    // is always `None` and unused.
    #[cfg(not(feature = "restricted"))]
    let _ = locked_policy_hashes;
    // Workers subscribe directly to graph store broadcasts —
    // no SyncManager needed.

    // ── Credential store + transform executor (full build only) ──
    // The restricted build does no server-side credential injection — a client
    // injects credentials itself from the transform_ids CheckToolCall returns —
    // so this whole stack (and the sasy-credential/rusqlite dep) is gated out.
    #[cfg(feature = "proxy")]
    let credential_source = build_credential_source(&credentials).await?;
    #[cfg(feature = "proxy")]
    let transform_config = if let Some(ref path) = transforms_file {
        TransformConfig::load(path).context("failed to load transforms")?
    } else {
        TransformConfig::default()
    };
    #[cfg(feature = "proxy")]
    let transform_executor = Arc::new(TransformExecutor::new(
        transform_config,
        Arc::clone(&credential_source),
    ));

    // ── Policy adapter for refmon ────────────────
    let policy_adapter = Arc::new(PolicyAdapter {
        engine: Arc::clone(&engine),
        graph_store: Arc::clone(&graph),
    });

    // Clone for forward proxy before Arcs move into services
    #[cfg(feature = "proxy")]
    let proxy_transform_executor = Arc::clone(&transform_executor);

    // ── gRPC services ────────────────────────────
    let obs_svc = ObservabilityService::new(Arc::clone(&graph));
    // The ObservabilityUpdates stream (live graph-update subscription for
    // external observers) is stripped from the restricted build, whose clients
    // use RegisterEventsWithDependencies + BackwardSlice only.
    #[cfg(not(feature = "restricted"))]
    let updates_svc = UpdatesService::new(Arc::clone(&graph));
    // Rehydrate persisted policy bindings before serving requests.
    // The lazy path (default) just refills the in-memory
    // `session_to_policy` + tenant-defaults maps from RocksDB —
    // O(1) at boot regardless of how many sessions were persisted.
    // The actual compile + install happens on the first dispatch
    // for each restored session (`ensure_installed`), amortizing
    // the cost across the request-arrival pattern instead of
    // paying it up front. Operators that need first-request
    // latency rather than fast startup can opt into the eager
    // path with `SASY_REPLAY_EAGER=1`.
    let replay_result = if sasy_policy::replay::eager_replay_requested() {
        // The eager path compiles at boot, so it is a load of persisted
        // functor source and takes the same gate as the lazy one.
        sasy_policy::replay::replay_persisted_policies_eager(
            &graph,
            engine.as_ref(),
            &policy_service_config,
        )
    } else {
        sasy_policy::replay::rehydrate_persisted_bindings(&graph, engine.as_ref())
    };
    match replay_result {
        Ok(stats) if stats.bindings_restored > 0 || stats.defaults_restored > 0 => {
            info!(?stats, "restored persisted policy state");
        }
        Ok(_) => {
            // Cold start with no persisted state — nothing to log.
        }
        Err(e) => {
            // Fatal. Replay is all-or-nothing: both scans have to succeed
            // before anything is installed, so one unreadable row drops every
            // session pin and every tenant default. Continuing then serves
            // tenant "default" from the `--policy` bootstrap — which the
            // documented `make serve POLICY_FILE=...` flow makes an ordinary
            // operator choice — so a session that was pinned to a stricter
            // policy comes back enforced by whatever that file says, with a
            // healthy health endpoint and one warning in the log. Other
            // tenants get "no policy registered" on every call, which is at
            // least loud.
            //
            // Refusing to start is the fail-closed answer and the visible one:
            // the operator sees why, and no request is served under a policy
            // nobody chose.
            return Err(anyhow::anyhow!(
                "policy replay failed, refusing to serve: {e}. The persisted \
                 bindings and tenant defaults could not be read, so sessions \
                 would resume under the bootstrap policy rather than the ones \
                 they were pinned to."
            ));
        }
    }

    let policy_svc = PolicyService::with_persistence(Arc::clone(&engine), Arc::clone(&graph))
        .with_config(policy_service_config);
    // Restricted builds reject custom policy uploads (curated profiles only)
    // and lock the bindable policy to the baked default selected at startup.
    #[cfg(feature = "restricted")]
    let policy_svc = policy_svc
        .with_restricted(true)
        .with_policy_lock(locked_policy_hashes);
    #[cfg(feature = "proxy")]
    let mut refmon_svc = RefmonService::new(policy_adapter, transform_executor);
    #[cfg(not(feature = "proxy"))]
    let mut refmon_svc = RefmonService::new(policy_adapter);
    if let Some(cfg) = auth_config {
        refmon_svc = refmon_svc.with_auth_config(cfg);
    }
    if trust_proxy_headers {
        refmon_svc = refmon_svc.trust_proxy_headers();
        info!("Trust proxy headers enabled (x-entity/x-roles accepted from any caller)");
    }
    #[cfg(feature = "proxy")]
    let credential_svc = CredentialService::new(credential_source);

    let socket_addr: SocketAddr = addr.parse().context("invalid listen address")?;

    // Restricted release builds serve loopback only — never expose the policy
    // engine on a routable interface.
    #[cfg(feature = "restricted")]
    if !socket_addr.ip().is_loopback() {
        anyhow::bail!(
            "restricted build refuses to bind non-loopback address {socket_addr}; use 127.0.0.1"
        );
    }

    // ── Forward proxy (optional; stripped in restricted/no-proxy builds) ──
    #[cfg(feature = "proxy")]
    if proxy_port > 0 {
        let proxy_addr: SocketAddr = format!("{}:{}", proxy_bind, proxy_port)
            .parse()
            .context("invalid proxy port")?;
        // Loud, because the proxy authenticates nobody and injects the
        // tenant's credentials: anything that can reach this address can
        // spend them against every host the policy allows.
        if !proxy_addr.ip().is_loopback() {
            tracing::warn!(
                %proxy_addr,
                "forward proxy is bound off-loopback; it authenticates no caller and \
                 attaches this tenant's stored credentials to allowed requests"
            );
        }
        // Clone before policy_adapter/transform_executor move into RefmonService
        let proxy_pc = Arc::new(PolicyAdapter {
            engine: Arc::clone(&engine),
            graph_store: Arc::clone(&graph),
        });
        let proxy_te = Arc::clone(&proxy_transform_executor);
        let proxy_tenant_for_listener = proxy_tenant.clone();
        tokio::spawn(async move {
            if let Err(e) = sasy_refmon::forward_proxy::run_forward_proxy(
                sasy_refmon::forward_proxy::ForwardProxyConfig {
                    listen_addr: proxy_addr,
                    tenant: proxy_tenant_for_listener,
                },
                proxy_pc,
                proxy_te,
            )
            .await
            {
                tracing::error!("forward proxy error: {}", e);
            }
        });
    }

    info!(%socket_addr, "starting gRPC server");

    // 16 MB — enough for large API responses proxied
    // through Qdrant, FDA, etc.
    const MAX_MSG: usize = 16 * 1024 * 1024;

    let mut builder = Server::builder();
    if let Some(ref tls) = tls_config {
        let tls_cfg = tls
            .load_server_config()
            .context("failed to load TLS config")?;
        builder = builder
            .tls_config(tls_cfg)
            .context("failed to configure TLS")?;
    }

    let router = builder.add_service(InterceptedService::new(
        ObservabilityServer::new(obs_svc)
            .max_decoding_message_size(MAX_MSG)
            .max_encoding_message_size(MAX_MSG),
        auth_interceptor.clone(),
    ));

    // ObservabilityUpdates (GetState + the bidirectional StreamUpdates graph
    // subscription) serves external live-graph observers — a graph browser or
    // Neo4j-style sync. Restricted builds have no such observer, so the stream
    // service is stripped from them.
    #[cfg(not(feature = "restricted"))]
    let router = router.add_service(InterceptedService::new(
        ObservabilityUpdatesServer::new(updates_svc)
            .max_decoding_message_size(MAX_MSG)
            .max_encoding_message_size(MAX_MSG),
        auth_interceptor.clone(),
    ));

    let router = router
        .add_service(InterceptedService::new(
            PolicyEngineServer::new(policy_svc)
                .max_decoding_message_size(MAX_MSG)
                .max_encoding_message_size(MAX_MSG),
            auth_interceptor.clone(),
        ))
        .add_service(InterceptedService::new(
            RmProxyServer::new(refmon_svc)
                .max_decoding_message_size(MAX_MSG)
                .max_encoding_message_size(MAX_MSG),
            auth_interceptor.clone(),
        ));

    // Credential server is stripped when the proxy/credential stack is gated
    // out (the restricted release build).
    #[cfg(feature = "proxy")]
    let router = router.add_service(InterceptedService::new(
        CredentialServerServer::new(credential_svc)
            .max_decoding_message_size(MAX_MSG)
            .max_encoding_message_size(MAX_MSG),
        auth_interceptor,
    ));

    router
        .serve(socket_addr)
        .await
        .context("gRPC server failed")?;

    Ok(())
}

// ── Credential population ───────────────────────────

/// Parse a `.env`-style file into key=value pairs.
///
/// Skips blank lines, comments (`#`), and lines without
/// `=`. Values are trimmed but not unquoted.
fn parse_env_file(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

/// The process environment, as name/value pairs, skipping anything that is
/// not valid UTF-8.
///
/// Not `std::env::vars()`: that one PANICS on a variable whose name or value
/// is not valid Unicode, and the panic message carries the offending value
/// through `Debug`. One stray variable — a binary password, a Latin-1 locale
/// value — would therefore stop the whole binary from starting and print a
/// secret into the log stream on the way out. A variable this convention
/// cannot read is not a credential it could have used anyway, so it is
/// dropped, by name, and nothing else is said about it.
#[cfg(feature = "proxy")]
fn readable_env_vars() -> Vec<(String, String)> {
    readable_vars(std::env::vars_os())
}

/// The UTF-8 pairs of `vars`; see [`readable_env_vars`].
#[cfg(feature = "proxy")]
fn readable_vars<I>(vars: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
{
    vars.into_iter()
        .filter_map(
            |(name, value)| match (name.into_string(), value.into_string()) {
                (Ok(name), Ok(value)) => Some((name, value)),
                (Ok(name), Err(_)) => {
                    warn!(
                        var = %name,
                        "environment variable is not valid UTF-8; not read as a credential"
                    );
                    None
                }
                // The name itself is unreadable, so there is nothing safe to
                // print: a lossy rendering of it is still the operator's data.
                (Err(_), _) => None,
            },
        )
        .collect()
}

/// The variables the memory backend is seeded from: the process environment,
/// then the entries of an optional `.env`-style file.
///
/// The order IS the precedence rule. Seeding applies the pairs in sequence and
/// each write overwrites the last, so a file the operator named explicitly
/// wins over whatever happens to be exported into the process — which is the
/// only reason to name one.
#[cfg(feature = "proxy")]
fn credential_seed_vars<I>(env: I, env_file_content: Option<&str>) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut vars: Vec<(String, String)> = env.into_iter().collect();
    if let Some(content) = env_file_content {
        vars.extend(parse_env_file(content));
    }
    vars
}

/// Populate the credential store from an env file.
///
/// Reads the file and applies the same environment-variable convention the
/// memory backend uses, so a name means the same thing whichever backend
/// reads it.
#[cfg(feature = "proxy")]
fn populate_credentials_from_env(store: &CredentialStore, env_file: &str) -> Result<usize> {
    let content = std::fs::read_to_string(env_file)
        .with_context(|| format!("failed to read env file: {env_file}"))?;
    let pairs = parse_env_file(&content);
    let count = sasy_credential::memory::seed_store_from_env_vars(store, pairs)?;
    Ok(count)
}

#[cfg(feature = "proxy")]
fn init_credentials(db_path: String, env_file: Option<String>) -> Result<()> {
    // A one-shot developer command, run in a checkout, so it keeps its
    // `data/credentials.db` default — but the path it acts on is absolute, and
    // that absolute path is what its errors and its confirmation name.
    let cwd = std::env::current_dir().context("failed to read the working directory")?;
    let db_path = cwd.join(&db_path);
    if let Some(parent) = db_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create credentials dir {}", parent.display()))?;
    }
    let db_path = db_path.to_string_lossy().into_owned();
    let store = CredentialStore::new(&db_path)
        .with_context(|| format!("failed to open credentials database {db_path}"))?;
    println!("credentials database initialized at {db_path}");
    info!("credentials database initialized at {db_path}");

    if let Some(ref path) = env_file {
        let n = populate_credentials_from_env(&store, path)?;
        info!(path, count = n, "populated credentials from env file");
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Where the credential database lives ──────

    /// Unset, the file hangs off the data directory — which `serve` has already
    /// made absolute — and never off the working directory.
    #[test]
    fn an_unset_credentials_db_hangs_off_the_data_directory() {
        let cwd = std::path::Path::new("/some/where/else");
        assert_eq!(
            resolve_credentials_db(None, "/srv/sasy/data", cwd),
            std::path::PathBuf::from("/srv/sasy/data/credentials.db")
        );
    }

    /// A relative value is the operator's, resolved against the directory the
    /// process started in; an absolute one is taken as written.
    #[test]
    fn a_given_credentials_db_resolves_against_the_startup_directory() {
        let cwd = std::path::Path::new("/home/dev/checkout");
        assert_eq!(
            resolve_credentials_db(Some("data/credentials.db"), "/srv/sasy/data", cwd),
            std::path::PathBuf::from("/home/dev/checkout/data/credentials.db")
        );
        assert_eq!(
            resolve_credentials_db(Some("/var/lib/sasy/creds.db"), "/srv/sasy/data", cwd),
            std::path::PathBuf::from("/var/lib/sasy/creds.db")
        );
    }

    /// The backends that keep nothing on disk create nothing on disk.
    #[cfg(feature = "proxy")]
    #[test]
    fn only_the_sqlite_backend_creates_a_credentials_directory() {
        let home = tempfile::TempDir::new().unwrap();
        let db = home.path().join("creds").join("credentials.db");

        for backend in [CredentialBackend::Memory, CredentialBackend::Openbao] {
            assert_eq!(credentials_dir_to_create(backend, &db), None);
            ensure_credentials_dir(backend, &db).expect("no directory is asked for");
            assert!(
                !db.parent().unwrap().exists(),
                "{backend:?} must not create a credentials directory"
            );
        }

        ensure_credentials_dir(CredentialBackend::Sqlite, &db).expect("sqlite creates its dir");
        assert!(db.parent().unwrap().is_dir());
    }

    /// A boot that dies creating the credential directory says which directory.
    #[cfg(feature = "proxy")]
    #[test]
    fn an_unwritable_credentials_parent_is_named_in_the_error() {
        use std::os::unix::fs::PermissionsExt;

        let home = tempfile::TempDir::new().unwrap();
        let locked = home.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o500)).unwrap();

        let db = locked.join("creds").join("credentials.db");
        let error = ensure_credentials_dir(CredentialBackend::Sqlite, &db)
            .expect_err("the parent cannot be created");
        let named = format!("{error:#}");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).unwrap();

        assert!(
            named.contains(&locked.join("creds").display().to_string()),
            "the error must name the directory: {named}"
        );
    }

    // ── Credential settings that go nowhere ──────

    fn default_credential_config() -> CredentialConfig {
        CredentialConfig {
            backend: CredentialBackend::Sqlite,
            db_path: "/srv/sasy/data/credentials.db".to_string(),
            db_path_set: false,
            env_file: None,
            openbao_addr: None,
            openbao_mount: "secret".to_string(),
            openbao_path_prefix: "sasy".to_string(),
            openbao_namespace: None,
            openbao_token_file: None,
            openbao_approle_role_id: None,
            openbao_approle_secret_id_file: None,
            openbao_ca_cert: None,
            openbao_cache_ttl_secs: 60,
            openbao_timeout_secs: 5,
        }
    }

    /// The env-file setting is read only by the memory backend, so with any
    /// other backend the operator is told it did nothing.
    #[test]
    fn the_env_file_is_reported_ignored_on_every_backend_but_memory() {
        let mut config = default_credential_config();
        assert!(
            !env_file_is_ignored(&config),
            "an unset env file is nothing to report"
        );
        config.env_file = Some("/tmp/creds.env".to_string());
        assert!(env_file_is_ignored(&config), "sqlite never reads it");
        config.backend = CredentialBackend::Openbao;
        assert!(env_file_is_ignored(&config), "openbao never reads it");
        config.backend = CredentialBackend::Memory;
        assert!(
            !env_file_is_ignored(&config),
            "the memory backend does seed from it"
        );
    }

    /// Settings left at their defaults are not reported: the restricted build
    /// would otherwise warn about flags nobody set.
    #[test]
    fn untouched_credential_settings_are_not_reported() {
        assert!(credential_flags_set(&default_credential_config()).is_empty());
    }

    /// Every credential setting the operator can give a value of their own is
    /// named. The restricted build parses all of them and builds no credential
    /// stack, so this list is what its startup warning prints.
    #[test]
    fn every_credential_setting_given_a_value_is_named() {
        let config = CredentialConfig {
            backend: CredentialBackend::Openbao,
            db_path: "/tmp/other.db".to_string(),
            db_path_set: true,
            env_file: Some("/tmp/creds.env".to_string()),
            openbao_addr: Some("https://bao.example:8200".to_string()),
            openbao_mount: "kv".to_string(),
            openbao_path_prefix: "apps".to_string(),
            openbao_namespace: Some("team".to_string()),
            openbao_token_file: Some("/tmp/token".to_string()),
            openbao_approle_role_id: Some("role".to_string()),
            openbao_approle_secret_id_file: Some("/tmp/secret-id".to_string()),
            openbao_ca_cert: Some("/tmp/ca.pem".to_string()),
            openbao_cache_ttl_secs: 120,
            openbao_timeout_secs: 9,
        };
        let set = credential_flags_set(&config);
        assert_eq!(
            set.len(),
            13,
            "every credential setting must be reported: {set:?}"
        );
        for flag in [
            "--credential-backend",
            "--credentials-db",
            "--credentials-env-file",
            "--openbao-addr",
            "--openbao-mount",
            "--openbao-path-prefix",
            "--openbao-namespace",
            "--openbao-token-file",
            "--openbao-approle-role-id",
            "--openbao-approle-secret-id-file",
            "--openbao-ca-cert",
            "--openbao-cache-ttl-secs",
            "--openbao-timeout-secs",
        ] {
            assert!(set.contains(&flag), "{flag} is missing from {set:?}");
        }
    }

    /// The env-file setting alone is enough to be named — it is the one that
    /// arrives from the environment on a deployment that never chose the
    /// memory backend.
    #[test]
    fn the_env_file_setting_alone_is_named() {
        let mut config = default_credential_config();
        config.env_file = Some("/tmp/creds.env".to_string());
        assert_eq!(
            credential_flags_set(&config),
            vec!["--credentials-env-file"]
        );
    }

    // ── The oracle-redaction switch ──────────────

    /// The flag is the operator's most recent word, so it decides even when
    /// the environment says the opposite; with no flag the environment
    /// decides; with neither, the strings are scrubbed.
    #[test]
    fn the_flag_beats_the_environment_and_absence_leaves_it_on() {
        assert!(
            oracle_redaction_setting(None, None),
            "nothing set: the oracle strings must still be scrubbed"
        );
        assert!(!oracle_redaction_setting(Some("off"), None));
        assert!(!oracle_redaction_setting(None, Some("off")));
        assert!(
            !oracle_redaction_setting(Some("off"), Some("on")),
            "the flag wins over the environment"
        );
        assert!(
            oracle_redaction_setting(Some("on"), Some("off")),
            "the flag wins over the environment in the other direction too"
        );
        assert!(
            !oracle_redaction_setting(Some(" OFF "), None),
            "the word is read whatever its case and spacing"
        );
    }

    /// A value that is neither word is not an answer. It is not read as `off`,
    /// and a flag that cannot be read does not let the environment's `off`
    /// through either: the switch stays on, which is the reading that sends
    /// nothing in the clear.
    #[test]
    fn a_value_the_binary_cannot_read_leaves_the_switch_on() {
        assert!(oracle_redaction_setting(Some("maybe"), None));
        assert!(oracle_redaction_setting(None, Some("0")));
        assert!(oracle_redaction_setting(None, Some("")));
        assert!(oracle_redaction_setting(Some("disabled"), Some("off")));
    }

    /// The flag is optional and spelled as the table documents it.
    #[test]
    fn the_oracle_redaction_flag_parses_and_defaults_to_absent() {
        let cli = Cli::try_parse_from(["observability", "serve"]).unwrap();
        match cli.command {
            Command::Serve(args) => assert_eq!(args.oracle_redaction, None),
            _ => panic!("expected Serve command"),
        }
        let cli =
            Cli::try_parse_from(["observability", "serve", "--oracle-redaction", "off"]).unwrap();
        match cli.command {
            Command::Serve(args) => assert_eq!(args.oracle_redaction.as_deref(), Some("off")),
            _ => panic!("expected Serve command"),
        }
    }

    // ── Gap 3: default listen address ────────────

    #[test]
    fn default_addr_is_ipv4_localhost() {
        // Parse CLI with no --addr override
        let cli = Cli::try_parse_from(["observability", "serve"]).unwrap();
        match cli.command {
            Command::Serve(args) => {
                let addr = args.addr;
                assert_eq!(
                    addr, "127.0.0.1:10089",
                    "default addr should be IPv4 localhost"
                );
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn custom_addr_override() {
        let cli =
            Cli::try_parse_from(["observability", "serve", "--addr", "0.0.0.0:9090"]).unwrap();
        match cli.command {
            Command::Serve(args) => {
                let addr = args.addr;
                assert_eq!(addr, "0.0.0.0:9090");
            }
            _ => panic!("expected Serve command"),
        }
    }

    /// The functor gate's opt-in reaches the policy service from either of the
    /// operator's two inputs, with the three values it can take, and is off
    /// when neither is given. This is the wiring, not the gate: the gate is
    /// tested in `sasy-policy` against a hand-built config, so a flag that
    /// never arrives — a dropped field, a misspelled env var — would leave
    /// every one of those tests passing while the shipped binary ignored the
    /// operator.
    #[test]
    fn the_functor_opt_in_reaches_the_policy_service_from_flag_or_env() {
        let cfg = |flag, env| policy_service_config(flag, env).map(|c| c.user_functors);
        assert_eq!(
            cfg(None, None),
            Ok(UserFunctors::Refuse),
            "off by default: functor source is native code in the engine's process"
        );
        assert_eq!(cfg(Some("sandboxed"), None), Ok(UserFunctors::Sandboxed));
        assert_eq!(
            cfg(Some("unsandboxed"), None),
            Ok(UserFunctors::Unsandboxed)
        );
        for on in ["1", "true", "yes", "on", "YES", "sandboxed"] {
            assert_eq!(
                cfg(None, Some(on)),
                Ok(UserFunctors::Sandboxed),
                "SASY_ALLOW_USER_FUNCTORS={on} should select the sandboxed opt-in"
            );
        }
        assert_eq!(
            cfg(None, Some("unsandboxed")),
            Ok(UserFunctors::Unsandboxed)
        );
        for off in ["0", "", "no", "off", "false"] {
            assert_eq!(
                cfg(None, Some(off)),
                Ok(UserFunctors::Refuse),
                "SASY_ALLOW_USER_FUNCTORS={off} should leave it off"
            );
        }
        for garbage in ["sandbox", "unsandbox", "maybe", "2"] {
            assert!(
                cfg(None, Some(garbage)).is_err(),
                "SASY_ALLOW_USER_FUNCTORS={garbage} must stop the boot, not resolve to \
                 one of the three answers by resembling it"
            );
        }
        // The flag wins over an inherited environment.
        assert_eq!(
            cfg(Some("unsandboxed"), Some("sandboxed")),
            Ok(UserFunctors::Unsandboxed)
        );
        // But the environment is still read when the flag is present, so a
        // value nobody can interpret stops the boot either way.
        assert!(
            cfg(Some("sandboxed"), Some("maybe")).is_err(),
            "an uninterpretable SASY_ALLOW_USER_FUNCTORS must stop the boot even when \
             --allow-user-functors would have overridden it"
        );
        assert!(
            policy_service_config(Some("sandboxed"), None)
                .unwrap()
                .sandbox_available
                .is_none(),
            "the sandbox answer is the host's, asked when the question is asked, not a \
             startup snapshot"
        );
    }

    /// The flag's three spellings on the command line: absent, bare, and each
    /// explicit value. A bare `--allow-user-functors` is the sandboxed opt-in,
    /// the conservative of the two.
    #[test]
    fn the_functor_flag_takes_an_optional_value() {
        let parse = |args: &[&str]| -> Option<String> {
            let mut argv = vec!["observability", "serve"];
            argv.extend_from_slice(args);
            match Cli::try_parse_from(argv).unwrap().command {
                Command::Serve(args) => args.allow_user_functors,
                #[allow(unreachable_patterns)]
                _ => panic!("expected Serve command"),
            }
        };
        assert_eq!(parse(&[]), None);
        assert_eq!(parse(&["--allow-user-functors"]), Some("sandboxed".into()));
        assert_eq!(
            parse(&["--allow-user-functors=sandboxed"]),
            Some("sandboxed".into())
        );
        assert_eq!(
            parse(&["--allow-user-functors=unsandboxed"]),
            Some("unsandboxed".into())
        );
        assert!(
            Cli::try_parse_from([
                "sasy",
                "serve",
                "--allow-user-functors=whenever-it-suits-you",
            ])
            .is_err(),
            "an unknown value must be refused by the parser, not interpreted"
        );
        // A common launch-script shape: the bare flag with the
        // next flag right behind it. An optional-value flag that swallowed
        // `--addr` would silently move the listener.
        match Cli::try_parse_from([
            "sasy",
            "serve",
            "--allow-user-functors",
            "--addr",
            "0.0.0.0:10089",
        ])
        .unwrap()
        .command
        {
            Command::Serve(args) => {
                assert_eq!(args.allow_user_functors, Some("sandboxed".into()));
                assert_eq!(args.addr, "0.0.0.0:10089");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected Serve command"),
        }
    }

    /// The unconfined-functor warning is one warning, in one situation: the
    /// operator asked for the unsandboxed opt-in on a host that has no
    /// sandbox. The other three combinations say nothing extra.
    #[test]
    fn the_unconfined_warning_fires_only_when_unsandboxed_and_there_is_no_sandbox() {
        assert_eq!(
            unconfined_functor_warning(UserFunctors::Unsandboxed, false)
                .map(|w| w.contains("unconfined")),
            Some(true)
        );
        assert!(unconfined_functor_warning(UserFunctors::Unsandboxed, true).is_none());
        assert!(unconfined_functor_warning(UserFunctors::Sandboxed, false).is_none());
        assert!(unconfined_functor_warning(UserFunctors::Sandboxed, true).is_none());
        assert!(unconfined_functor_warning(UserFunctors::Refuse, false).is_none());
        assert!(unconfined_functor_warning(UserFunctors::Refuse, true).is_none());
    }

    // ── Gap 2: env file parsing ─────────────────

    #[test]
    fn parse_env_file_basic() {
        let content = "\
OPENAI_API_KEY=sk-test-123
OPENFDA_API_KEY=fda-key-456
";
        let pairs = parse_env_file(content);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "OPENAI_API_KEY");
        assert_eq!(pairs[0].1, "sk-test-123");
        assert_eq!(pairs[1].0, "OPENFDA_API_KEY");
        assert_eq!(pairs[1].1, "fda-key-456");
    }

    #[test]
    fn parse_env_file_with_comments_and_blanks() {
        let content = "\
# This is a comment
OPENAI_API_KEY=sk-test

  # another comment

OPENFDA_API_KEY=fda-key
";
        let pairs = parse_env_file(content);
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn parse_env_file_empty() {
        let pairs = parse_env_file("");
        assert!(pairs.is_empty());
    }

    #[test]
    fn parse_env_file_no_equals() {
        let content = "BROKEN_LINE\n";
        let pairs = parse_env_file(content);
        assert!(pairs.is_empty());
    }

    #[cfg(feature = "proxy")]
    #[test]
    fn populate_credentials_from_env_file() {
        let store = CredentialStore::new(":memory:").unwrap();

        // Write a temp env file
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".env-credentials");
        std::fs::write(
            &env_path,
            "OPENAI_API_KEY=sk-test-123\n\
             OPENFDA_API_KEY=fda-key-456\n",
        )
        .unwrap();

        let count = populate_credentials_from_env(&store, env_path.to_str().unwrap()).unwrap();
        assert_eq!(count, 2);

        // Verify OpenAI credential
        let creds = store.get_credentials("*", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "sk-test-123");

        // Verify FDA credential
        let creds = store.get_credentials("*", "fda").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "fda-key-456");

        // Wildcard lookup works for any entity
        let creds = store.get_credentials("service-client", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "sk-test-123");
    }

    /// `--credentials-env-file` is documented as winning over the
    /// environment, which is the whole reason to pass it: the operator names
    /// a file to override whatever the process happens to have inherited.
    /// Seeding applies the pairs in order, so the file's entries have to come
    /// last.
    #[cfg(feature = "proxy")]
    #[tokio::test]
    async fn a_credentials_env_file_overrides_the_same_variable_in_the_environment() {
        let vars = credential_seed_vars(
            [
                ("OPENAI_API_KEY".to_string(), "sk-from-environment".to_string()),
                ("BRAVE_API_KEY".to_string(), "brave-from-environment".to_string()),
            ],
            Some("OPENAI_API_KEY=sk-from-file\n# a comment\nSASY_CREDENTIAL_STRIPE_SECRET_KEY=stripe-from-file\n"),
        );

        let source = sasy_credential::MemorySource::new().unwrap();
        let count = source.seed(vars).unwrap();
        assert_eq!(count, 4, "one variable is seeded twice, the file's last");

        let resolved = |service: &'static str| {
            let source = &source;
            async move {
                source
                    .resolve(
                        "default",
                        "agent",
                        service,
                        sasy_credential::CredentialView::Relay,
                    )
                    .await
                    .unwrap()
                    .values
            }
        };

        assert_eq!(
            resolved("openai").await.get("api_key").map(String::as_str),
            Some("sk-from-file"),
            "the named file must win over the environment"
        );
        assert_eq!(
            resolved("brave").await.get("api_key").map(String::as_str),
            Some("brave-from-environment"),
            "a variable the file says nothing about is still seeded"
        );
        assert_eq!(
            resolved("stripe")
                .await
                .get("secret_key")
                .map(String::as_str),
            Some("stripe-from-file"),
            "the generic convention applies to file entries too"
        );
    }

    /// One unreadable variable must not stop the binary from starting, and
    /// nothing of what it held may be printed.
    ///
    /// `std::env::vars()` panics on a name or value that is not valid
    /// Unicode, and formats the offending `OsString` into the panic message —
    /// so a single binary-valued variable in the environment (which is where
    /// this backend's credentials live) would both abort startup and put that
    /// value in the log stream.
    #[cfg(all(feature = "proxy", unix))]
    #[test]
    fn an_environment_variable_that_is_not_utf8_is_skipped_rather_than_fatal() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let readable = readable_vars(vec![
            (
                OsString::from("OPENAI_API_KEY"),
                OsString::from("sk-readable"),
            ),
            (
                OsString::from("SASY_CREDENTIAL_STRIPE_SECRET_KEY"),
                // Not valid UTF-8: a lone continuation byte.
                OsString::from_vec(vec![b's', b'k', 0x80]),
            ),
            (
                OsString::from_vec(vec![0xff, b'N', b'A', b'M', b'E']),
                OsString::from("unreadable name"),
            ),
        ]);

        assert_eq!(
            readable,
            vec![("OPENAI_API_KEY".to_string(), "sk-readable".to_string())],
            "the readable variables are seeded and the rest are dropped"
        );
    }

    /// Where credentials come from defaults to the local SQLite file, and an
    /// operator can select a backend by flag or by environment variable.
    ///
    /// The env fallback is read off the parser rather than by setting a
    /// variable: the environment is shared by every test in this binary, and
    /// a test that mutates it decides what the others see.
    #[test]
    fn the_credential_backend_defaults_to_sqlite_and_reads_the_environment() {
        let backend_of = |args: &[&str]| match Cli::try_parse_from(args).unwrap().command {
            Command::Serve(args) => args.credential_backend,
            _ => panic!("expected Serve command"),
        };

        assert_eq!(
            backend_of(&["observability", "serve"]),
            CredentialBackend::Sqlite,
            "a deployment that chooses nothing keeps the local file it had"
        );
        for (value, expected) in [
            ("sqlite", CredentialBackend::Sqlite),
            ("memory", CredentialBackend::Memory),
            ("openbao", CredentialBackend::Openbao),
        ] {
            assert_eq!(
                backend_of(&["observability", "serve", "--credential-backend", value]),
                expected
            );
        }
        assert!(
            Cli::try_parse_from(["observability", "serve", "--credential-backend", "vault"])
                .is_err(),
            "an unknown backend name is refused rather than falling back"
        );

        use clap::CommandFactory;
        let serve = Cli::command()
            .find_subcommand("serve")
            .cloned()
            .expect("serve subcommand");
        let arg = serve
            .get_arguments()
            .find(|a| a.get_id() == "credential_backend")
            .expect("--credential-backend");
        assert_eq!(
            arg.get_env().and_then(|e| e.to_str()),
            Some("SASY_CREDENTIAL_BACKEND"),
            "the backend must also be selectable from the environment"
        );
    }

    /// A configuration with no OpenBao settings at all, for the tests below
    /// to spoil one field at a time.
    #[cfg(feature = "proxy")]
    fn openbao_config_missing_everything() -> CredentialConfig {
        CredentialConfig {
            backend: CredentialBackend::Openbao,
            db_path: "unused".to_string(),
            db_path_set: false,
            env_file: None,
            openbao_addr: None,
            openbao_mount: "secret".to_string(),
            openbao_path_prefix: "sasy".to_string(),
            openbao_namespace: None,
            openbao_token_file: None,
            openbao_approle_role_id: None,
            openbao_approle_secret_id_file: None,
            openbao_ca_cert: None,
            openbao_cache_ttl_secs: 60,
            openbao_timeout_secs: 5,
        }
    }

    /// The openbao backend needs an address and exactly one way to
    /// authenticate, and says so before it opens a socket.
    ///
    /// A half-specified configuration must be a refusal to start, not a
    /// server that comes up and denies every authorized request needing a
    /// credential. The both-supplied case matters most: it is the one where
    /// the process would otherwise silently ignore one of the two credential
    /// paths the operator configured.
    #[cfg(feature = "proxy")]
    #[tokio::test]
    async fn openbao_needs_an_address_and_exactly_one_way_to_authenticate() {
        let err = build_credential_source(&openbao_config_missing_everything())
            .await
            .map(|_| ())
            .expect_err("no address is a refusal to start");
        assert!(
            format!("{err}").contains("--openbao-addr"),
            "the error names what is missing: {err}"
        );

        let with_addr = || {
            let mut config = openbao_config_missing_everything();
            config.openbao_addr = Some("https://bao.example.com:8200".to_string());
            config
        };

        let both = {
            let mut config = with_addr();
            config.openbao_token_file = Some("/run/secrets/token".to_string());
            config.openbao_approle_role_id = Some("role-1".to_string());
            config.openbao_approle_secret_id_file = Some("/run/secrets/secret-id".to_string());
            config
        };
        let role_id_only = {
            let mut config = with_addr();
            config.openbao_approle_role_id = Some("role-1".to_string());
            config
        };
        let secret_id_only = {
            let mut config = with_addr();
            config.openbao_approle_secret_id_file = Some("/run/secrets/secret-id".to_string());
            config
        };

        let cases = [
            ("no authentication at all", with_addr()),
            ("a token file AND an AppRole", both),
            ("a role id with no secret id file", role_id_only),
            ("a secret id file with no role id", secret_id_only),
        ];
        for (what, config) in cases {
            let err = match build_credential_source(&config).await {
                Ok(_) => panic!("{what} must be refused before the process starts"),
                Err(e) => e,
            };
            assert!(
                format!("{err}").contains("exactly one"),
                "{what}: the refusal must say what a valid configuration looks like: {err}"
            );
        }
    }

    #[cfg(feature = "proxy")]
    #[test]
    fn populate_credentials_missing_file_errors() {
        let store = CredentialStore::new(":memory:").unwrap();
        let result = populate_credentials_from_env(&store, "/nonexistent/.env-credentials");
        assert!(result.is_err());
    }

    #[cfg(feature = "proxy")]
    #[test]
    fn populate_credentials_partial_env() {
        let store = CredentialStore::new(":memory:").unwrap();

        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".env-credentials");
        std::fs::write(
            &env_path,
            "OPENAI_API_KEY=sk-only\n\
             UNRELATED_VAR=ignored\n",
        )
        .unwrap();

        let count = populate_credentials_from_env(&store, env_path.to_str().unwrap()).unwrap();
        assert_eq!(count, 1, "only OPENAI_API_KEY mapped");

        let creds = store.get_credentials("*", "openai").unwrap();
        assert_eq!(creds.get("api_key").unwrap(), "sk-only");

        // FDA should be empty
        let creds = store.get_credentials("*", "fda").unwrap();
        assert!(creds.is_empty());
    }

    // ── Gap 1: --auth-provider CLI flag ─────────

    #[test]
    fn auth_provider_flag_parsed() {
        let cli = Cli::try_parse_from([
            "sasy",
            "serve",
            "--auth-provider",
            "config/auth/apikey.json",
        ])
        .unwrap();
        match cli.command {
            Command::Serve(args) => {
                let auth_provider = args.auth_provider;
                assert_eq!(auth_provider.as_deref(), Some("config/auth/apikey.json"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn auth_provider_flag_defaults_to_none() {
        let cli = Cli::try_parse_from(["observability", "serve"]).unwrap();
        match cli.command {
            Command::Serve(args) => {
                let auth_provider = args.auth_provider;
                assert!(auth_provider.is_none());
            }
            _ => panic!("expected Serve command"),
        }
    }

    // ── init-credentials --env-file flag ────────

    #[cfg(feature = "proxy")]
    #[test]
    fn init_credentials_env_file_flag_parsed() {
        let cli =
            Cli::try_parse_from(["sasy", "init-credentials", "--env-file", ".env-credentials"])
                .unwrap();
        match cli.command {
            Command::InitCredentials { env_file, .. } => {
                assert_eq!(env_file.as_deref(), Some(".env-credentials"));
            }
            _ => panic!("expected InitCredentials"),
        }
    }

    #[cfg(feature = "proxy")]
    #[test]
    fn init_credentials_env_file_defaults_to_none() {
        let cli = Cli::try_parse_from(["observability", "init-credentials"]).unwrap();
        match cli.command {
            Command::InitCredentials { env_file, .. } => {
                assert!(env_file.is_none());
            }
            _ => panic!("expected InitCredentials"),
        }
    }

    // ── --policy-plugin CLI flag ────────────────

    #[test]
    fn policy_plugin_flag_parsed() {
        let cli = Cli::try_parse_from([
            "sasy",
            "serve",
            "--policy-plugin",
            "/path/to/libsasy_policy.so",
        ])
        .unwrap();
        match cli.command {
            Command::Serve(args) => {
                let policy_plugin = args.policy_plugin;
                assert_eq!(policy_plugin.as_deref(), Some("/path/to/libsasy_policy.so"));
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn policy_plugin_flag_defaults_to_none() {
        let cli = Cli::try_parse_from(["observability", "serve"]).unwrap();
        match cli.command {
            Command::Serve(args) => {
                let policy_plugin = args.policy_plugin;
                assert!(policy_plugin.is_none());
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn policy_plugin_coexists_with_metadata() {
        let cli = Cli::try_parse_from([
            "sasy",
            "serve",
            "--policy-plugin",
            "/path/to/plugin.so",
            "--policy-metadata",
            "policy.dl",
        ])
        .unwrap();
        match cli.command {
            Command::Serve(args) => {
                assert!(args.policy_plugin.is_some());
                assert!(args.policy_metadata.is_some());
            }
            _ => panic!("expected Serve command"),
        }
    }
}

//! Standalone reference monitor gRPC service.
//!
//! Connects to a remote policy engine via gRPC for authorization,
//! and a credential service for credential injection.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use sasy_common::services::rm_proxy_server::RmProxyServer;
use sasy_credential::SqliteSource;
use sasy_refmon::policy::{endpoint_is_loopback, GrpcPolicyChecker};
use sasy_refmon::{RefmonService, TransformConfig, TransformExecutor};
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "refmon-service", about = "Reference monitor gRPC service")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:50054")]
    addr: String,

    /// URL of the policy engine gRPC service
    #[arg(long, default_value = "http://localhost:50053")]
    policy_engine: String,

    /// Transforms JSON file
    #[arg(long)]
    transforms: Option<String>,

    /// Auth provider JSON config (for gRPC RBAC)
    #[arg(long)]
    auth_provider: Option<String>,

    /// Auth config YAML (entity→role mappings)
    #[arg(long)]
    auth_config: Option<String>,

    /// Credential DB path
    #[arg(long, default_value = "data/credentials.db")]
    credentials_db: String,

    /// HTTP forward proxy port (0 to disable)
    #[arg(long, default_value = "0")]
    proxy_port: u16,

    /// Address the forward proxy listens on. Loopback by default: the proxy
    /// authenticates nobody and attaches the tenant's stored credentials to
    /// requests the policy allows, so anything that can reach it can spend
    /// those credentials against every host the policy permits. Binding it
    /// off-host is a deliberate act.
    #[arg(long, default_value = "127.0.0.1")]
    proxy_bind: String,

    /// Tenant the forward proxy listener routes traffic into.
    /// Multi-tenant operators run one proxy per tenant.
    #[arg(long, default_value = "default")]
    proxy_tenant: String,

    #[arg(long)]
    tls_cert: Option<String>,
    #[arg(long)]
    tls_key: Option<String>,
    #[arg(long)]
    tls_ca: Option<String>,

    /// Client mTLS cert presented to the *policy engine*, so the
    /// engine binds this refmon to a `service-proxy` identity and
    /// honors its per-tenant/principal delegation. Required (with
    /// --engine-tls-key) for a non-loopback --policy-engine unless
    /// --engine-api-key is set instead.
    #[arg(long)]
    engine_tls_cert: Option<String>,
    #[arg(long)]
    engine_tls_key: Option<String>,
    /// CA the engine's server cert is verified against.
    #[arg(long)]
    engine_tls_ca: Option<String>,
    /// Server-name override for the engine's TLS cert (when its
    /// CN/SAN doesn't match the endpoint host, e.g. dev certs whose
    /// CN is "reference-monitor").
    #[arg(long)]
    engine_tls_domain: Option<String>,
    /// API key presented to the engine as `x-api-key` (alternative to
    /// engine mTLS for the refmon's service-proxy identity). Falls
    /// back to $SASY_ENGINE_API_KEY.
    #[arg(long)]
    engine_api_key: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let addr: SocketAddr = cli.addr.parse().context("parse address")?;

    // Credential source. The standalone reference monitor reads the local
    // SQLite file; the backend choice lives on `sasy serve`.
    if let Some(parent) = std::path::Path::new(&cli.credentials_db).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let credential_source = Arc::new(SqliteSource::open(&cli.credentials_db)?);

    // Transform executor
    let transform_config = if let Some(ref path) = cli.transforms {
        TransformConfig::load(path).context("load transforms")?
    } else {
        TransformConfig::default()
    };
    let transform_executor = Arc::new(TransformExecutor::new(transform_config, credential_source));

    // Engine-client credentials. In a split deployment the refmon
    // must authenticate to the engine; otherwise the engine cannot
    // bind it to a `service-proxy` identity and will reject (or, on a
    // misconfigured trust-all engine, blindly accept) its per-end-user
    // tenant/principal delegation — the latter a tenant-spoofing
    // vector.
    let engine_tls = if cli.engine_tls_cert.is_some()
        || cli.engine_tls_key.is_some()
        || cli.engine_tls_ca.is_some()
    {
        let tcfg = sasy_auth::TlsConfig {
            cert_path: cli.engine_tls_cert.clone(),
            key_path: cli.engine_tls_key.clone(),
            ca_path: cli.engine_tls_ca.clone(),
        };
        let mut ctls = tcfg
            .load_client_config()
            .context("build engine client TLS config")?;
        if let Some(domain) = &cli.engine_tls_domain {
            ctls = ctls.domain_name(domain.clone());
        }
        Some(ctls)
    } else {
        None
    };
    let engine_api_key = cli
        .engine_api_key
        .clone()
        .or_else(|| std::env::var("SASY_ENGINE_API_KEY").ok());
    // A client *identity* (mTLS cert+key) or an API key authenticates
    // the refmon. A bare CA (server-auth only) does not.
    let has_engine_identity =
        (cli.engine_tls_cert.is_some() && cli.engine_tls_key.is_some()) || engine_api_key.is_some();

    if !has_engine_identity {
        if endpoint_is_loopback(&cli.policy_engine) {
            tracing::warn!(
                policy_engine = %cli.policy_engine,
                "engine connection is unauthenticated; the refmon's tenant/principal delegation \
                 relies on the engine trusting same-host callers. Set --engine-tls-cert/\
                 --engine-tls-key (mTLS) or --engine-api-key for a hardened split deployment."
            );
        } else {
            anyhow::bail!(
                "refusing to start: --policy-engine {} is not loopback but no engine-client \
                 credentials are set. The refmon must authenticate to the engine \
                 (--engine-tls-cert + --engine-tls-key [+ --engine-tls-ca] for mTLS, or \
                 --engine-api-key) so the engine can bind it to a service-proxy identity and \
                 honor its per-tenant delegation. Without this the engine rejects the delegation, \
                 or — if it trusts unauthenticated callers — lets any client reaching it assert \
                 arbitrary tenants.",
                cli.policy_engine
            );
        }
    }

    // Policy checker (remote gRPC)
    let policy_checker = Arc::new(GrpcPolicyChecker::with_auth(
        &cli.policy_engine,
        engine_tls,
        engine_api_key,
    ));

    // Forward proxy (optional)
    if cli.proxy_port > 0 {
        let proxy_addr: std::net::SocketAddr = format!("{}:{}", cli.proxy_bind, cli.proxy_port)
            .parse()
            .context("proxy port")?;
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
        let proxy_pc = Arc::clone(&policy_checker);
        let proxy_te = Arc::clone(&transform_executor);
        let proxy_tenant = cli.proxy_tenant.clone();
        tokio::spawn(async move {
            if let Err(e) = sasy_refmon::forward_proxy::run_forward_proxy(
                sasy_refmon::forward_proxy::ForwardProxyConfig {
                    listen_addr: proxy_addr,
                    tenant: proxy_tenant,
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

    // Auth config
    let mut refmon_svc = RefmonService::new(policy_checker, transform_executor);
    if let Some(ref path) = cli.auth_config {
        let cfg = sasy_auth::AuthConfig::load(path).context("load auth config")?;
        refmon_svc = refmon_svc.with_auth_config(cfg);
    }

    // Auth interceptor for gRPC RBAC
    let ac2 = if let Some(ref p) = cli.auth_config {
        Some(sasy_auth::AuthConfig::load(p).context("load auth config for interceptor")?)
    } else {
        None
    };
    let ap: std::sync::Arc<dyn sasy_auth::AuthProvider> = if let Some(ref p) = cli.auth_provider {
        sasy_auth::load_provider_from_file(std::path::Path::new(p), ac2)
            .context("load auth provider")?
    } else {
        std::sync::Arc::new(sasy_auth::PassthroughAuthProvider::new(ac2))
    };
    let auth_interceptor = sasy_auth::AuthInterceptor::new(ap);

    info!(%addr, policy_engine = %cli.policy_engine, "starting reference monitor");

    let mut builder = Server::builder();
    if let (Some(cert), Some(key)) = (&cli.tls_cert, &cli.tls_key) {
        let tls = sasy_auth::TlsConfig::new(cert, key, cli.tls_ca.as_deref().unwrap_or(""));
        builder = builder.tls_config(tls.load_server_config()?)?;
        info!("TLS enabled");
    }

    builder
        .add_service(tonic::service::interceptor::InterceptedService::new(
            RmProxyServer::new(refmon_svc),
            auth_interceptor,
        ))
        .serve(addr)
        .await?;

    Ok(())
}

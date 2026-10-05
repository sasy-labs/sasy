//! Standalone observability + updates gRPC service.
//!
//! Runs GraphStore (RocksDB) + ObservabilityService + UpdatesService.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use sasy_common::observability::{
    observability_server::ObservabilityServer,
    observability_updates_server::ObservabilityUpdatesServer,
};
use sasy_graph::GraphStore;
use sasy_server::{ObservabilityService, UpdatesService};
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "sasy-server", about = "Observability graph + updates service")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:50052")]
    addr: String,

    #[arg(long, default_value = "data/graph")]
    data_dir: String,

    /// Auth provider JSON config file
    #[arg(long)]
    auth_provider: Option<String>,

    /// Auth config YAML (entity→role mappings)
    #[arg(long)]
    auth_config: Option<String>,

    /// TLS server certificate path
    #[arg(long)]
    tls_cert: Option<String>,

    /// TLS server private key path
    #[arg(long)]
    tls_key: Option<String>,

    /// TLS CA certificate path (enables mTLS)
    #[arg(long)]
    tls_ca: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    std::fs::create_dir_all(&cli.data_dir).context("create data dir")?;
    let graph = Arc::new(GraphStore::new(Some(&cli.data_dir)).context("open graph store")?);

    // Auth
    let auth_config = if let Some(ref path) = cli.auth_config {
        Some(sasy_auth::AuthConfig::load(path).context("load auth config")?)
    } else {
        None
    };
    let auth_provider: std::sync::Arc<dyn sasy_auth::AuthProvider> =
        if let Some(ref path) = cli.auth_provider {
            sasy_auth::load_provider_from_file(std::path::Path::new(path), auth_config.clone())
                .context("load auth provider")?
        } else {
            std::sync::Arc::new(sasy_auth::PassthroughAuthProvider::new(auth_config.clone()))
        };
    let auth_interceptor = sasy_auth::AuthInterceptor::new(auth_provider);

    let obs_svc = ObservabilityService::new(Arc::clone(&graph));
    let updates_svc = UpdatesService::new(Arc::clone(&graph));

    let addr: SocketAddr = cli.addr.parse().context("parse address")?;
    info!(%addr, "starting observability server");

    let mut builder = Server::builder();
    if let (Some(cert), Some(key)) = (&cli.tls_cert, &cli.tls_key) {
        let tls = sasy_auth::TlsConfig::new(cert, key, cli.tls_ca.as_deref().unwrap_or(""));
        let tls_cfg = tls.load_server_config().context("load TLS")?;
        builder = builder.tls_config(tls_cfg).context("configure TLS")?;
        info!("TLS enabled");
    }

    builder
        .add_service(InterceptedService::new(
            ObservabilityServer::new(obs_svc),
            auth_interceptor.clone(),
        ))
        .add_service(InterceptedService::new(
            ObservabilityUpdatesServer::new(updates_svc),
            auth_interceptor,
        ))
        .serve(addr)
        .await?;

    Ok(())
}

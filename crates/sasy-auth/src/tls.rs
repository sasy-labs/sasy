//! TLS configuration utilities for tonic servers and
//! clients.

use std::fs;

use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

use crate::error::AuthError;

/// TLS file paths for certificates and keys.
#[derive(Debug, Clone, Default)]
pub struct TlsConfig {
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub ca_path: Option<String>,
}

impl TlsConfig {
    pub fn new(cert: impl Into<String>, key: impl Into<String>, ca: impl Into<String>) -> Self {
        Self {
            cert_path: Some(cert.into()),
            key_path: Some(key.into()),
            ca_path: Some(ca.into()),
        }
    }

    /// Build a tonic `ServerTlsConfig`.
    pub fn load_server_config(&self) -> Result<ServerTlsConfig, AuthError> {
        let cert = self.read_cert()?;
        let key = self.read_key()?;
        let identity = Identity::from_pem(cert, key);

        let mut tls = ServerTlsConfig::new().identity(identity);

        if let Some(ref ca_path) = self.ca_path {
            let ca =
                fs::read(ca_path).map_err(|e| AuthError::Tls(format!("read CA {ca_path}: {e}")))?;
            tls = tls.client_ca_root(Certificate::from_pem(ca));
        }

        Ok(tls)
    }

    /// Build a tonic `ClientTlsConfig`.
    pub fn load_client_config(&self) -> Result<ClientTlsConfig, AuthError> {
        let mut tls = ClientTlsConfig::new();

        if let Some(ref ca_path) = self.ca_path {
            let ca =
                fs::read(ca_path).map_err(|e| AuthError::Tls(format!("read CA {ca_path}: {e}")))?;
            tls = tls.ca_certificate(Certificate::from_pem(ca));
        }

        if self.cert_path.is_some() && self.key_path.is_some() {
            let cert = self.read_cert()?;
            let key = self.read_key()?;
            tls = tls.identity(Identity::from_pem(cert, key));
        }

        Ok(tls)
    }

    /// Whether both client cert and key are set (mTLS).
    pub fn is_mtls(&self) -> bool {
        self.cert_path.is_some() && self.key_path.is_some() && self.ca_path.is_some()
    }

    /// Whether any TLS paths are configured.
    pub fn is_configured(&self) -> bool {
        self.cert_path.is_some() || self.key_path.is_some() || self.ca_path.is_some()
    }

    fn read_cert(&self) -> Result<Vec<u8>, AuthError> {
        let path = self
            .cert_path
            .as_ref()
            .ok_or_else(|| AuthError::Tls("no cert_path set".into()))?;
        fs::read(path).map_err(|e| AuthError::Tls(format!("read cert {path}: {e}")))
    }

    fn read_key(&self) -> Result<Vec<u8>, AuthError> {
        let path = self
            .key_path
            .as_ref()
            .ok_or_else(|| AuthError::Tls("no key_path set".into()))?;
        fs::read(path).map_err(|e| AuthError::Tls(format!("read key {path}: {e}")))
    }
}

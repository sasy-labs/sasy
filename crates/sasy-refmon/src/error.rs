//! Error types for the reference monitor crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RefmonError {
    #[error("policy check failed: {0}")]
    PolicyCheck(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("transform error: {0}")]
    Transform(String),

    #[error("proxy error: {0}")]
    Proxy(String),

    #[error("config error: {0}")]
    Config(String),

    #[cfg(feature = "proxy")]
    #[error("credential error: {0}")]
    Credential(#[from] sasy_credential::CredentialError),

    #[cfg(feature = "proxy")]
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<RefmonError> for tonic::Status {
    fn from(err: RefmonError) -> Self {
        match &err {
            RefmonError::PolicyCheck(_) => tonic::Status::permission_denied(err.to_string()),
            _ => tonic::Status::internal(err.to_string()),
        }
    }
}

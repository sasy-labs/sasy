//! Error types for the authentication crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("authentication failed: {0}")]
    Unauthenticated(String),

    #[error("invalid token: {0}")]
    InvalidToken(String),

    #[error("token expired")]
    TokenExpired,

    #[error("config error: {0}")]
    Config(String),

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<AuthError> for tonic::Status {
    fn from(err: AuthError) -> Self {
        match &err {
            AuthError::Unauthenticated(_) => tonic::Status::unauthenticated(err.to_string()),
            AuthError::InvalidToken(_) | AuthError::TokenExpired => {
                tonic::Status::unauthenticated(err.to_string())
            }
            AuthError::Config(_) => tonic::Status::internal(err.to_string()),
            AuthError::Tls(_) => tonic::Status::internal(err.to_string()),
            AuthError::Io(_) => tonic::Status::internal(err.to_string()),
        }
    }
}

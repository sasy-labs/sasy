//! Error types for the credential store.

/// Errors that can occur in credential operations.
#[derive(Debug, thiserror::Error)]
pub enum CredentialError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// The store was asked to open something it cannot secure — a SQLite URI
    /// or a symlink — where continuing would mean holding credentials in a
    /// file whose permissions are not ours to set. Also covers a backend
    /// configured in a way that cannot be honoured.
    #[error("credential store configuration: {0}")]
    Config(String),

    /// A remote credential backend could not be reached, or answered in a way
    /// that is not a credential. Never carries a response body: bodies from a
    /// secret manager can contain secrets.
    #[error("credential backend: {0}")]
    Backend(String),

    /// The configured backend does not accept writes. The message names the
    /// backend and where it reads from, so an operator is told where the
    /// write actually belongs.
    #[error("{0}")]
    ReadOnly(String),
}

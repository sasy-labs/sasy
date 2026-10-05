//! Error types for the graph store.

/// Errors that can occur in graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("RocksDB error: {0}")]
    Rocks(#[from] rocksdb::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Invalid edge: {0}")]
    InvalidEdge(String),

    #[error("Invalid immutable snapshot: {0}")]
    InvalidSnapshot(String),

    #[error("Immutable message: {0}")]
    ImmutableViolation(String),

    /// The store on disk was written in a format this build does not read.
    /// Carries the whole explanation, path and versions included, because it
    /// is surfaced to an operator at startup and nothing downstream can add
    /// to it.
    #[error("{0}")]
    SchemaVersion(String),

    /// A write to a content-addressed key found DIFFERENT bytes already
    /// stored under it. Two distinct contents claiming one hash is either a
    /// hash collision or a caller writing under a key it did not derive from
    /// the bytes it is writing; either way the stored record is left alone
    /// and the write is refused. Carries the whole explanation because the
    /// caller has nothing to add to it.
    #[error("{0}")]
    ContentHashAmbiguity(String),
}

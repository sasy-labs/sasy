//! Embedded graph store using petgraph + RocksDB.
//!
//! An in-process [`GraphStore`] that persists to RocksDB
//! and emits change events via `tokio::sync::broadcast`.

pub mod convert;
pub mod error;
#[cfg(feature = "neo4j-sync")]
pub mod neo4j_sync;
pub mod persistence;
pub mod store;
pub mod types;

pub use error::GraphError;
pub use persistence::{
    ClearedBindingMetadata, FunctorAdmission, PersistedPolicy, PolicyMetadataFact,
};
pub use store::{event_carries_nothing, GraphStore, OwnerClaim, OwnerClaimToken};
pub use types::GraphUpdate;

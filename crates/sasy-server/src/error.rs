//! Server error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("graph error: {0}")]
    Graph(#[from] sasy_graph::GraphError),

    #[error("missing required field: {0}")]
    MissingField(&'static str),

    #[error("internal: {0}")]
    Internal(String),
}

impl From<ServerError> for tonic::Status {
    fn from(e: ServerError) -> Self {
        match e {
            ServerError::MissingField(f) => {
                tonic::Status::invalid_argument(format!("missing field: {f}"))
            }
            ServerError::Graph(sasy_graph::GraphError::NodeNotFound(id)) => {
                tonic::Status::not_found(format!("node not found: {id}"))
            }
            other => tonic::Status::internal(other.to_string()),
        }
    }
}

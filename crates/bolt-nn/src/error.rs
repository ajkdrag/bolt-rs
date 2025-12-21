use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("state error: {0}")]
    State(String),
    #[error("io error: {0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, Error>;


use thiserror::Error;

pub type Result<T> = std::result::Result<T, HeartError>;

#[derive(Debug, Error)]
pub enum HeartError {
    #[error("database error: {0}")]
    Database(#[from] redb::Error),
    #[error("database open/create error: {0}")]
    DatabaseOpen(#[from] redb::DatabaseError),
    #[error("database transaction error: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("database table error: {0}")]
    Table(#[from] redb::TableError),
    #[error("database storage error: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("database commit error: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("cryptographic operation failed")]
    Crypto,
    #[error("invalid recovery phrase")]
    InvalidRecoveryPhrase,
    #[error("invalid passphrase or corrupt store header")]
    UnlockFailed,
    #[error("store schema {found} is not supported; expected {expected}")]
    UnsupportedSchema { found: u32, expected: u32 },
    #[error("event signature is invalid")]
    InvalidSignature,
    #[error("event identity does not match its signed body")]
    InvalidEventId,
    #[error("event already exists with different content")]
    EventCollision,
    #[error("requested object was not found")]
    NotFound,
    #[error("snapshot is read-only")]
    ReadOnly,
    #[error("cognitive projection is stale and must be rebuilt before incremental updates")]
    ProjectionStale,
    #[error("model error: {0}")]
    Model(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<postcard::Error> for HeartError {
    fn from(value: postcard::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

impl From<serde_json::Error> for HeartError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#![forbid(unsafe_code)]

mod crypto;
mod dcmdb;
mod embedding;
mod error;
mod event;
mod facts;
mod heart;
mod ids;
mod projection;
mod risk;
mod store;
mod sync;
mod thymos;
mod triangle;
mod vector;
mod verifier;

pub use crypto::{CreatedKeys, KeySource, RecoveryPhrase};
pub use dcmdb::{
    Dcmdb, DcmdbConfig, MaintenanceReport, MemoryNode, MemoryObservation, RecallHit, TensionInfo,
    kappa_from_m,
};
pub use embedding::{Embedding, ModelManifest, SemanticEncoder};
pub use error::{HeartError, Result};
pub use event::{
    AttachmentRef, ColdBlob, ColdBlobRef, Content, EventBody, EventKind, HybridTimestamp,
    InteractionInput, ParticipantRole, Provenance, SignedEvent, ToolExchange,
};
pub use facts::{
    Fact, FactAggregation, FactCandidate, FactExtractor, FactHit, FactSlotType, FactStore,
    FactValue, TimeSource,
};
pub use heart::{
    CommitReceipt, CreatedHeart, HeartConfig, ReadOnlyHeart, RecalledMemory, SpineHeart,
};
pub use ids::{
    AgentId, BlobId, DeviceId, EventId, FactId, NodeId, SnapshotId, ThreadId, TombstoneId,
    TriangleId,
};
pub use projection::{CognitiveConfig, CognitiveState, MemoryReceipt};
pub use risk::RiskField;
pub use store::{DeviceAuthorization, Snapshot, StoreStats, Tombstone, TombstoneTarget};
pub use sync::{EncryptedDelta, ImportReceipt, SyncFrontier};
pub use thymos::{
    ActivationNonlinearity, FeelingVector, Thymos, ThymosConfig, TrajectoryStep, ValenceMode,
};
pub use triangle::{
    ContextBranch, ContextForest, ContextHandle, ContextLeaf, ContextTriangle, CoordinateRole,
    RehydrateBudget, RehydratedContext, ResolvedCoordinate, TriangleConfig,
};
pub use verifier::{
    AtomicClaim, ClaimExtractor, ClaimRelation, NliLabelOrder, NliModel, NliProbabilities,
    NliReport, NliVerifier,
};

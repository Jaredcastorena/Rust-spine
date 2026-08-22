use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AgentId, BlobId, DeviceId, EventId, ThreadId};

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct HybridTimestamp {
    pub wall_millis: u64,
    pub counter: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParticipantRole {
    User,
    Assistant,
    Tool,
    Operator,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventKind {
    Message,
    ToolCall,
    ToolResult,
    Outcome,
    Reflection,
    AgentPromoted,
    Control,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ColdBlobRef {
    pub id: BlobId,
    pub media_type: String,
    pub plaintext_len: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ColdBlob {
    pub reference: ColdBlobRef,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Content {
    Inline(String),
    ColdBlob(ColdBlobRef),
    Redacted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttachmentRef {
    pub name: String,
    pub blob: ColdBlobRef,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub source_uri: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolExchange {
    pub operation_id: String,
    pub tool_name: String,
    pub arguments: Content,
    pub result: Option<Content>,
    pub succeeded: Option<bool>,
    pub background: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InteractionInput {
    pub agent_id: AgentId,
    pub thread_id: ThreadId,
    pub role: ParticipantRole,
    pub kind: EventKind,
    pub content: Content,
    pub causal_parents: Vec<EventId>,
    pub provenance: Provenance,
    pub tool: Option<ToolExchange>,
    pub attachments: Vec<AttachmentRef>,
    pub outcome: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventBody {
    pub schema: u32,
    pub device_id: DeviceId,
    pub authorization_epoch: u64,
    pub device_sequence: u64,
    pub timestamp: HybridTimestamp,
    pub interaction: InteractionInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignedEvent {
    pub id: EventId,
    pub body: EventBody,
    pub signer_public_key: [u8; 32],
    pub signature: Vec<u8>,
}

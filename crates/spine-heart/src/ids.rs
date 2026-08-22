use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{HeartError, Result};

macro_rules! hash_id {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
            Deserialize,
        )]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const fn from_bytes(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&hex::encode(self.0))
            }
        }

        impl FromStr for $name {
            type Err = HeartError;

            fn from_str(value: &str) -> Result<Self> {
                let bytes = hex::decode(value).map_err(|_| {
                    HeartError::InvalidInput(concat!("invalid ", stringify!($name)).into())
                })?;
                let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
                    HeartError::InvalidInput(
                        concat!("invalid ", stringify!($name), " length").into(),
                    )
                })?;
                Ok(Self(bytes))
            }
        }
    };
}

hash_id!(BlobId);
hash_id!(DeviceId);
hash_id!(EventId);
hash_id!(FactId);
hash_id!(NodeId);
hash_id!(SnapshotId);
hash_id!(TombstoneId);
hash_id!(TriangleId);

macro_rules! named_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                let trimmed = value.trim();
                if trimmed.is_empty() || trimmed.len() > 256 {
                    return Err(HeartError::InvalidInput(
                        concat!(stringify!($name), " must contain 1..=256 characters").into(),
                    ));
                }
                Ok(Self(trimmed.to_owned()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

named_id!(AgentId);
named_id!(ThreadId);

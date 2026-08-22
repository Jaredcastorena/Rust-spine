use std::collections::{BTreeMap, BTreeSet};

use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, KeyInit, Payload},
};
use serde::{Deserialize, Serialize};

use crate::{
    BlobId, ColdBlob, DeviceAuthorization, DeviceId, HeartError, Result, SignedEvent, Snapshot,
    SnapshotId, Tombstone, TombstoneId, store::Store,
};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SyncFrontier {
    pub devices: BTreeMap<DeviceId, u64>,
    pub snapshots: BTreeSet<SnapshotId>,
    pub tombstones: BTreeSet<TombstoneId>,
    pub blobs: BTreeSet<BlobId>,
    pub authorizations: BTreeMap<DeviceId, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EncryptedDelta {
    pub schema: u32,
    pub nonce: [u8; 24],
    pub ciphertext: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImportReceipt {
    pub received: u64,
    pub inserted: u64,
    pub duplicates: u64,
    pub snapshots: u64,
    pub tombstones: u64,
    pub blobs: u64,
    pub authorizations: u64,
}

#[derive(Serialize, Deserialize)]
struct DeltaBody {
    schema: u32,
    events: Vec<SignedEvent>,
    snapshots: Vec<Snapshot>,
    tombstones: Vec<Tombstone>,
    blobs: Vec<ColdBlob>,
    authorizations: Vec<DeviceAuthorization>,
}

pub(crate) fn export_delta(store: &Store, remote: &SyncFrontier) -> Result<EncryptedDelta> {
    let body = DeltaBody {
        schema: 1,
        events: store.events_after(&remote.devices)?,
        snapshots: store
            .snapshots()?
            .into_iter()
            .filter(|snapshot| !remote.snapshots.contains(&snapshot.id))
            .collect(),
        tombstones: store
            .tombstones()?
            .into_iter()
            .filter(|tombstone| !remote.tombstones.contains(&tombstone.id))
            .collect(),
        blobs: store
            .blobs()?
            .into_iter()
            .filter(|blob| !remote.blobs.contains(&blob.reference.id))
            .collect(),
        authorizations: store
            .authorizations()?
            .into_iter()
            .filter(|authorization| {
                authorization.epoch
                    > remote
                        .authorizations
                        .get(&authorization.device_id)
                        .copied()
                        .unwrap_or_default()
            })
            .collect(),
    };
    let bytes = postcard::to_allocvec(&body)?;
    let cipher = XChaCha20Poly1305::new(store.sync_key().into());
    let mut nonce_bytes = [0_u8; 24];
    getrandom::fill(&mut nonce_bytes).map_err(|_| HeartError::Crypto)?;
    let nonce = XNonce::from(nonce_bytes);
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: &bytes,
                aad: b"spine-sync-delta-v1",
            },
        )
        .map_err(|_| HeartError::Crypto)?;
    Ok(EncryptedDelta {
        schema: 1,
        nonce: nonce_bytes,
        ciphertext,
    })
}

pub(crate) fn import_delta(store: &Store, delta: EncryptedDelta) -> Result<ImportReceipt> {
    if delta.schema != 1 {
        return Err(HeartError::UnsupportedSchema {
            found: delta.schema,
            expected: 1,
        });
    }
    let cipher = XChaCha20Poly1305::new(store.sync_key().into());
    let nonce = XNonce::from(delta.nonce);
    let bytes = cipher
        .decrypt(
            &nonce,
            Payload {
                msg: &delta.ciphertext,
                aad: b"spine-sync-delta-v1",
            },
        )
        .map_err(|_| HeartError::Crypto)?;
    let body: DeltaBody = postcard::from_bytes(&bytes)?;
    if body.schema != 1 {
        return Err(HeartError::UnsupportedSchema {
            found: body.schema,
            expected: 1,
        });
    }
    let mut receipt = ImportReceipt {
        received: body.events.len() as u64,
        ..ImportReceipt::default()
    };
    for authorization in body.authorizations {
        store.put_authorization(&authorization)?;
        receipt.authorizations += 1;
    }
    for event in body.events {
        if store.put_event(&event)? {
            receipt.inserted += 1;
        } else {
            receipt.duplicates += 1;
        }
    }
    for snapshot in body.snapshots {
        store.put_snapshot(&snapshot)?;
        receipt.snapshots += 1;
    }
    for tombstone in body.tombstones {
        store.put_tombstone(&tombstone)?;
        receipt.tombstones += 1;
    }
    for blob in body.blobs {
        if store.put_blob_record(&blob, 1_048_576)? {
            receipt.blobs += 1;
        }
    }
    Ok(receipt)
}

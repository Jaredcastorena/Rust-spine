use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use ed25519_dalek::{Signer, SigningKey};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    BlobId, ColdBlob, ColdBlobRef, DeviceId, EventId, HeartError, Result, SignedEvent, SnapshotId,
    TombstoneId,
    crypto::{
        DeviceIdentity, KeySource, StoreKeys, WrappedRoot, decrypt_record, derive_store_keys,
        encrypt_record, generate_root, open_object, root_from_phrase, seal_object, unwrap_root,
        verify_detached, verify_event, wrap_root,
    },
};

const SCHEMA_VERSION: u32 = 3;
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");
const EVENTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("events");
const EVENT_ORDER: TableDefinition<&[u8], &[u8]> = TableDefinition::new("event_order");
const SNAPSHOTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("snapshots");
const TOMBSTONES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("tombstones");
const DATA_KEYS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("object_data_keys");
const PROJECTIONS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("projections");
const BLOBS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("blobs");
const AUTHORIZATIONS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("device_authorizations");

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoreHeader {
    schema: u32,
    owner_public_key: [u8; 32],
    wrapped_root: WrappedRoot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DeviceRecord {
    signing_secret: [u8; 32],
    next_sequence: u64,
    last_wall_millis: u64,
    hlc_counter: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceAuthorization {
    pub device_id: DeviceId,
    pub signing_public_key: [u8; 32],
    pub epoch: u64,
    pub owner_public_key: [u8; 32],
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub label: Option<String>,
    pub created_wall_millis: u64,
    pub event_frontier: BTreeMap<DeviceId, u64>,
    pub projection_generation: u64,
    pub model_manifest_hash: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TombstoneTarget {
    Event(EventId),
    Snapshot(SnapshotId),
    Blob(crate::BlobId),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Tombstone {
    pub id: TombstoneId,
    pub target: TombstoneTarget,
    pub device_id: DeviceId,
    pub authorization_epoch: u64,
    pub device_sequence: u64,
    pub wall_millis: u64,
    pub reason: Option<String>,
    pub signer_public_key: [u8; 32],
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StoreStats {
    pub events: u64,
    pub blobs: u64,
    pub snapshots: u64,
    pub tombstones: u64,
}

pub(crate) struct CreatedStore {
    pub store: Store,
    pub recovery_phrase: crate::RecoveryPhrase,
}

#[derive(Clone)]
pub(crate) struct Store {
    db: Arc<Database>,
    keys: Arc<StoreKeys>,
    device: Arc<std::sync::Mutex<DeviceIdentity>>,
    blob_dir: Arc<PathBuf>,
}

impl Store {
    pub fn create(path: &Path, passphrase: &str) -> Result<CreatedStore> {
        let created = generate_root()?;
        let store = Self::create_with_root(path, &created.root_key, passphrase)?;
        Ok(CreatedStore {
            store,
            recovery_phrase: created.recovery_phrase,
        })
    }

    pub fn create_replica(path: &Path, recovery_phrase: &str, passphrase: &str) -> Result<Self> {
        let root = root_from_phrase(recovery_phrase)?;
        Self::create_with_root(path, &root, passphrase)
    }

    fn create_with_root(path: &Path, root: &[u8; 32], passphrase: &str) -> Result<Self> {
        if path.exists() {
            return Err(HeartError::InvalidInput(format!(
                "refusing to overwrite existing store {}",
                path.display()
            )));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let keys = derive_store_keys(root)?;
        let owner_secret = derive_owner_secret(&keys)?;
        let owner_public_key = ed25519_dalek::SigningKey::from_bytes(&owner_secret)
            .verifying_key()
            .to_bytes();
        let header = StoreHeader {
            schema: SCHEMA_VERSION,
            owner_public_key,
            wrapped_root: wrap_root(root, passphrase)?,
        };
        let device = DeviceIdentity::generate()?;
        let device_record = DeviceRecord {
            signing_secret: *device.secret(),
            next_sequence: 1,
            last_wall_millis: 0,
            hlc_counter: 0,
        };
        let db = Arc::new(Database::create(path)?);
        let store = Self {
            db,
            keys: Arc::new(keys),
            device: Arc::new(std::sync::Mutex::new(device)),
            blob_dir: Arc::new(blob_directory(path)),
        };
        store.initialize(&header, &device_record)?;
        store.authorize_current_device(1)?;
        Ok(store)
    }

    pub fn open(path: &Path, source: KeySource) -> Result<Self> {
        let db = Arc::new(Database::open(path)?);
        let header = read_header(&db)?;
        if header.schema != SCHEMA_VERSION {
            return Err(HeartError::UnsupportedSchema {
                found: header.schema,
                expected: SCHEMA_VERSION,
            });
        }
        let root = match &source {
            KeySource::Passphrase(passphrase) => unwrap_root(&header.wrapped_root, passphrase)?,
            KeySource::RecoveryPhrase(phrase) => root_from_phrase(phrase)?,
            KeySource::RootKey(root) => zeroize::Zeroizing::new(*root),
        };
        let keys = Arc::new(derive_store_keys(&root)?);
        let record: DeviceRecord = read_encrypted_meta(&db, &keys, "device")?;
        let device = DeviceIdentity::from_secret(record.signing_secret);
        let store = Self {
            db,
            keys,
            device: Arc::new(std::sync::Mutex::new(device)),
            blob_dir: Arc::new(blob_directory(path)),
        };
        store.current_authorization_epoch()?;
        Ok(store)
    }

    fn initialize(&self, header: &StoreHeader, device: &DeviceRecord) -> Result<()> {
        let write = self.db.begin_write()?;
        {
            let mut meta = write.open_table(META)?;
            let header_bytes = serde_json::to_vec(header)?;
            meta.insert("header", header_bytes.as_slice())?;
            let device_bytes = postcard::to_allocvec(device)?;
            let encrypted = encrypt_record(&self.keys, b"meta:device", &device_bytes)?;
            meta.insert("device", encrypted.as_slice())?;
        }
        write.open_table(EVENTS)?;
        write.open_table(EVENT_ORDER)?;
        write.open_table(SNAPSHOTS)?;
        write.open_table(TOMBSTONES)?;
        write.open_table(DATA_KEYS)?;
        write.open_table(PROJECTIONS)?;
        write.open_table(BLOBS)?;
        write.open_table(AUTHORIZATIONS)?;
        write.commit()?;
        Ok(())
    }

    pub fn device_id(&self) -> DeviceId {
        self.device.lock().expect("device lock poisoned").id()
    }

    pub fn reserve_clock(
        &self,
        observed_wall_millis: u64,
    ) -> Result<(u64, crate::HybridTimestamp)> {
        let mut record: DeviceRecord = read_encrypted_meta(&self.db, &self.keys, "device")?;
        let wall = observed_wall_millis.max(record.last_wall_millis);
        let counter = if wall == record.last_wall_millis {
            record.hlc_counter.saturating_add(1)
        } else {
            0
        };
        let sequence = record.next_sequence;
        record.next_sequence = record.next_sequence.saturating_add(1);
        record.last_wall_millis = wall;
        record.hlc_counter = counter;
        write_encrypted_meta(&self.db, &self.keys, "device", &record)?;
        Ok((
            sequence,
            crate::HybridTimestamp {
                wall_millis: wall,
                counter,
            },
        ))
    }

    pub fn sign_event(&self, body: crate::EventBody) -> Result<SignedEvent> {
        self.device
            .lock()
            .expect("device lock poisoned")
            .sign_event(&self.keys, body)
    }

    pub fn current_authorization_epoch(&self) -> Result<u64> {
        let device_id = self.device_id();
        self.authorizations()?
            .into_iter()
            .filter(|authorization| authorization.device_id == device_id)
            .map(|authorization| authorization.epoch)
            .max()
            .ok_or(HeartError::InvalidSignature)
    }

    pub fn authorizations(&self) -> Result<Vec<DeviceAuthorization>> {
        self.all_encrypted(AUTHORIZATIONS)
    }

    pub fn put_authorization(&self, authorization: &DeviceAuthorization) -> Result<()> {
        self.verify_authorization(authorization)?;
        let key = self.authorization_key(authorization.device_id, authorization.epoch);
        self.put_encrypted(AUTHORIZATIONS, &key, authorization)
    }

    pub fn sign_bytes(&self, bytes: &[u8]) -> ([u8; 32], Vec<u8>) {
        self.device
            .lock()
            .expect("device lock poisoned")
            .sign_bytes(bytes)
    }

    pub fn put_event(&self, event: &SignedEvent) -> Result<bool> {
        verify_event(&self.keys, event)?;
        self.verify_authorized_record(
            event.body.device_id,
            event.body.authorization_epoch,
            &event.signer_public_key,
        )?;
        if self.is_tombstoned(&TombstoneTarget::Event(event.id))? {
            return Ok(false);
        }
        let key = event.id.as_bytes();
        let bytes = postcard::to_allocvec(event)?;
        let (wrapped_key, encrypted) = seal_object(&self.keys, key, &bytes)?;
        let order_key = event_order_key(event);

        let write = self.db.begin_write()?;
        let inserted = {
            let mut events = write.open_table(EVENTS)?;
            if let Some(existing) = events.get(key.as_slice())? {
                let existing = existing.value();
                let data_keys = write.open_table(DATA_KEYS)?;
                let wrapped = data_keys.get(key.as_slice())?.ok_or(HeartError::NotFound)?;
                let decoded = open_object(&self.keys, key, wrapped.value(), existing)?;
                if decoded != bytes {
                    return Err(HeartError::EventCollision);
                }
                false
            } else {
                events.insert(key.as_slice(), encrypted.as_slice())?;
                let mut data_keys = write.open_table(DATA_KEYS)?;
                data_keys.insert(key.as_slice(), wrapped_key.as_slice())?;
                let mut order = write.open_table(EVENT_ORDER)?;
                order.insert(order_key.as_slice(), key.as_slice())?;
                true
            }
        };
        write.commit()?;
        Ok(inserted)
    }

    pub fn get_event(&self, id: EventId) -> Result<Option<SignedEvent>> {
        let read = self.db.begin_read()?;
        let table = read.open_table(EVENTS)?;
        let Some(value) = table.get(id.as_bytes().as_slice())? else {
            return Ok(None);
        };
        let data_keys = read.open_table(DATA_KEYS)?;
        let Some(wrapped_key) = data_keys.get(id.as_bytes().as_slice())? else {
            return Ok(None);
        };
        let bytes = open_object(
            &self.keys,
            id.as_bytes(),
            wrapped_key.value(),
            value.value(),
        )?;
        let event: SignedEvent = postcard::from_bytes(&bytes)?;
        verify_event(&self.keys, &event)?;
        Ok(Some(event))
    }

    pub fn events_canonical(&self) -> Result<Vec<SignedEvent>> {
        let read = self.db.begin_read()?;
        let order = read.open_table(EVENT_ORDER)?;
        let events = read.open_table(EVENTS)?;
        let data_keys = read.open_table(DATA_KEYS)?;
        let mut result = Vec::new();
        for entry in order.iter()? {
            let (_, id) = entry?;
            let id = id.value();
            let Some(value) = events.get(id)? else {
                return Err(HeartError::NotFound);
            };
            let id_array: [u8; 32] = id.try_into().map_err(|_| HeartError::NotFound)?;
            let Some(wrapped_key) = data_keys.get(id)? else {
                continue;
            };
            let bytes = open_object(&self.keys, &id_array, wrapped_key.value(), value.value())?;
            let event: SignedEvent = postcard::from_bytes(&bytes)?;
            verify_event(&self.keys, &event)?;
            result.push(event);
        }
        Ok(result)
    }

    pub fn put_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        self.verify_snapshot(snapshot)?;
        self.put_encrypted(SNAPSHOTS, snapshot.id.as_bytes(), snapshot)
    }

    pub fn get_snapshot(&self, id: SnapshotId) -> Result<Option<Snapshot>> {
        self.get_encrypted(SNAPSHOTS, id.as_bytes())
    }

    pub fn put_tombstone(&self, tombstone: &Tombstone) -> Result<()> {
        self.verify_tombstone(tombstone)?;
        self.put_encrypted(TOMBSTONES, tombstone.id.as_bytes(), tombstone)?;
        self.crypto_shred(&tombstone.target)
    }

    pub fn snapshots(&self) -> Result<Vec<Snapshot>> {
        self.all_encrypted(SNAPSHOTS)
    }

    pub fn tombstones(&self) -> Result<Vec<Tombstone>> {
        self.all_encrypted(TOMBSTONES)
    }

    pub fn is_tombstoned(&self, target: &TombstoneTarget) -> Result<bool> {
        Ok(self
            .tombstones()?
            .into_iter()
            .any(|tombstone| &tombstone.target == target))
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let read = self.db.begin_read()?;
        Ok(StoreStats {
            events: self.events_canonical()?.len() as u64,
            blobs: self.blobs()?.len() as u64,
            snapshots: read.open_table(SNAPSHOTS)?.len()?,
            tombstones: read.open_table(TOMBSTONES)?.len()?,
        })
    }

    pub fn put_blob(
        &self,
        media_type: &str,
        bytes: &[u8],
        external_threshold: usize,
    ) -> Result<ColdBlobRef> {
        let id = self.blob_id(media_type, bytes)?;
        let reference = ColdBlobRef {
            id,
            media_type: media_type.to_owned(),
            plaintext_len: bytes.len() as u64,
        };
        self.put_blob_record(
            &ColdBlob {
                reference: reference.clone(),
                bytes: bytes.to_vec(),
            },
            external_threshold,
        )?;
        Ok(reference)
    }

    pub fn put_blob_record(&self, blob: &ColdBlob, external_threshold: usize) -> Result<bool> {
        if self.blob_id(&blob.reference.media_type, &blob.bytes)? != blob.reference.id
            || blob.reference.plaintext_len != blob.bytes.len() as u64
        {
            return Err(HeartError::InvalidInput(
                "invalid cold blob identity".into(),
            ));
        }
        if self.is_tombstoned(&TombstoneTarget::Blob(blob.reference.id))? {
            return Ok(false);
        }
        if let Some(existing) = self.get_blob(blob.reference.id)? {
            if existing == *blob {
                return Ok(false);
            }
            return Err(HeartError::EventCollision);
        }
        let id = blob.reference.id;
        let package = postcard::to_allocvec(blob)?;
        let (wrapped_key, encrypted) = seal_object(&self.keys, id.as_bytes(), &package)?;
        let external = blob.bytes.len() >= external_threshold;
        if external {
            fs::create_dir_all(self.blob_dir.as_path())?;
            let final_path = self.blob_path(id);
            if !final_path.exists() {
                let mut random = [0_u8; 8];
                getrandom::fill(&mut random).map_err(|_| HeartError::Crypto)?;
                let temporary = self
                    .blob_dir
                    .join(format!("{}.{}.tmp", id, hex::encode(random)));
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temporary)?;
                file.write_all(&encrypted)?;
                file.sync_all()?;
                fs::rename(temporary, final_path)?;
            }
        }
        let mut index_value = vec![u8::from(external)];
        if !external {
            index_value.extend_from_slice(&encrypted);
        }
        let write = self.db.begin_write()?;
        {
            let mut table = write.open_table(BLOBS)?;
            table.insert(id.as_bytes().as_slice(), index_value.as_slice())?;
            let mut data_keys = write.open_table(DATA_KEYS)?;
            data_keys.insert(id.as_bytes().as_slice(), wrapped_key.as_slice())?;
        }
        write.commit()?;
        Ok(true)
    }

    pub fn get_blob(&self, id: BlobId) -> Result<Option<ColdBlob>> {
        let read = self.db.begin_read()?;
        let blobs = read.open_table(BLOBS)?;
        let Some(index) = blobs.get(id.as_bytes().as_slice())? else {
            return Ok(None);
        };
        let index = index.value();
        let Some((&external, inline)) = index.split_first() else {
            return Err(HeartError::NotFound);
        };
        let data_keys = read.open_table(DATA_KEYS)?;
        let Some(wrapped_key) = data_keys.get(id.as_bytes().as_slice())? else {
            return Ok(None);
        };
        let external_bytes;
        let encrypted = if external == 1 {
            external_bytes = fs::read(self.blob_path(id))?;
            external_bytes.as_slice()
        } else if external == 0 {
            inline
        } else {
            return Err(HeartError::NotFound);
        };
        let bytes = open_object(&self.keys, id.as_bytes(), wrapped_key.value(), encrypted)?;
        let blob: ColdBlob = postcard::from_bytes(&bytes)?;
        if self.blob_id(&blob.reference.media_type, &blob.bytes)? != id {
            return Err(HeartError::InvalidInput(
                "cold blob identity mismatch".into(),
            ));
        }
        Ok(Some(blob))
    }

    pub fn blobs(&self) -> Result<Vec<ColdBlob>> {
        let ids = {
            let read = self.db.begin_read()?;
            let table = read.open_table(BLOBS)?;
            let mut ids = Vec::new();
            for entry in table.iter()? {
                let (id, _) = entry?;
                ids.push(BlobId::from_bytes(
                    id.value().try_into().map_err(|_| HeartError::NotFound)?,
                ));
            }
            ids
        };
        ids.into_iter()
            .filter_map(|id| self.get_blob(id).transpose())
            .collect()
    }

    pub fn put_projection<T: Serialize>(&self, generation: u64, value: &T) -> Result<()> {
        let key = self.projection_key(generation);
        self.put_encrypted(PROJECTIONS, &key, value)
    }

    pub fn get_projection<T: DeserializeOwned>(&self, generation: u64) -> Result<Option<T>> {
        let key = self.projection_key(generation);
        self.get_encrypted(PROJECTIONS, &key)
    }

    pub fn frontier(&self) -> Result<BTreeMap<DeviceId, u64>> {
        let mut frontier = BTreeMap::new();
        for event in self.events_canonical()? {
            frontier
                .entry(event.body.device_id)
                .and_modify(|value: &mut u64| *value = (*value).max(event.body.device_sequence))
                .or_insert(event.body.device_sequence);
        }
        Ok(frontier)
    }

    pub fn events_after(&self, frontier: &BTreeMap<DeviceId, u64>) -> Result<Vec<SignedEvent>> {
        Ok(self
            .events_canonical()?
            .into_iter()
            .filter(|event| {
                event.body.device_sequence
                    > frontier
                        .get(&event.body.device_id)
                        .copied()
                        .unwrap_or_default()
            })
            .collect())
    }

    pub fn sync_key(&self) -> &[u8; 32] {
        &self.keys.sync
    }

    pub fn object_id(&self, domain: &[u8], bytes: &[u8]) -> [u8; 32] {
        let mut input = Vec::with_capacity(domain.len() + bytes.len());
        input.extend_from_slice(domain);
        input.extend_from_slice(bytes);
        *blake3::keyed_hash(&self.keys.object_id, &input).as_bytes()
    }

    fn blob_id(&self, media_type: &str, bytes: &[u8]) -> Result<BlobId> {
        let identity = postcard::to_allocvec(&(media_type, bytes))?;
        Ok(BlobId::from_bytes(self.object_id(b"blob-v1", &identity)))
    }

    fn blob_path(&self, id: BlobId) -> PathBuf {
        self.blob_dir.join(format!("{id}.blob"))
    }

    fn projection_key(&self, generation: u64) -> [u8; 32] {
        self.object_id(b"cognitive-projection-v1", &generation.to_be_bytes())
    }

    pub fn snapshot_id(&self, snapshot: &Snapshot) -> Result<SnapshotId> {
        let bytes = postcard::to_allocvec(&(
            snapshot.created_wall_millis,
            &snapshot.event_frontier,
            &snapshot.label,
            snapshot.projection_generation,
        ))?;
        Ok(SnapshotId::from_bytes(self.object_id(b"snapshot", &bytes)))
    }

    pub fn tombstone_id(&self, tombstone: &Tombstone) -> Result<TombstoneId> {
        let bytes = tombstone_signing_bytes(tombstone)?;
        Ok(TombstoneId::from_bytes(
            self.object_id(b"tombstone", &bytes),
        ))
    }

    fn authorization_key(&self, device_id: DeviceId, epoch: u64) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(40);
        bytes.extend_from_slice(device_id.as_bytes());
        bytes.extend_from_slice(&epoch.to_be_bytes());
        self.object_id(b"device-authorization-v1", &bytes)
    }

    fn authorize_current_device(&self, epoch: u64) -> Result<()> {
        let device = self.device.lock().expect("device lock poisoned");
        let owner = SigningKey::from_bytes(&derive_owner_secret(&self.keys)?);
        let mut authorization = DeviceAuthorization {
            device_id: device.id(),
            signing_public_key: device.public(),
            epoch,
            owner_public_key: owner.verifying_key().to_bytes(),
            signature: Vec::new(),
        };
        let bytes = authorization_signing_bytes(&authorization)?;
        authorization.signature = owner.sign(&bytes).to_bytes().to_vec();
        drop(device);
        self.put_authorization(&authorization)
    }

    fn verify_authorization(&self, authorization: &DeviceAuthorization) -> Result<()> {
        if authorization.epoch == 0
            || DeviceId(*blake3::hash(&authorization.signing_public_key).as_bytes())
                != authorization.device_id
        {
            return Err(HeartError::InvalidSignature);
        }
        let expected_owner = SigningKey::from_bytes(&derive_owner_secret(&self.keys)?)
            .verifying_key()
            .to_bytes();
        if authorization.owner_public_key != expected_owner {
            return Err(HeartError::InvalidSignature);
        }
        verify_detached(
            &authorization.owner_public_key,
            &authorization_signing_bytes(authorization)?,
            &authorization.signature,
        )
    }

    fn verify_authorized_record(
        &self,
        device_id: DeviceId,
        epoch: u64,
        signing_public_key: &[u8; 32],
    ) -> Result<()> {
        let authorization = self
            .authorizations()?
            .into_iter()
            .find(|item| item.device_id == device_id && item.epoch == epoch)
            .ok_or(HeartError::InvalidSignature)?;
        self.verify_authorization(&authorization)?;
        if authorization.signing_public_key != *signing_public_key {
            return Err(HeartError::InvalidSignature);
        }
        Ok(())
    }

    fn verify_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        if self.snapshot_id(snapshot)? != snapshot.id {
            return Err(HeartError::InvalidInput("invalid snapshot identity".into()));
        }
        Ok(())
    }

    fn verify_tombstone(&self, tombstone: &Tombstone) -> Result<()> {
        let bytes = tombstone_signing_bytes(tombstone)?;
        if self.tombstone_id(tombstone)? != tombstone.id {
            return Err(HeartError::InvalidInput(
                "invalid tombstone identity".into(),
            ));
        }
        let signer_device = DeviceId(*blake3::hash(&tombstone.signer_public_key).as_bytes());
        if signer_device != tombstone.device_id {
            return Err(HeartError::InvalidSignature);
        }
        self.verify_authorized_record(
            tombstone.device_id,
            tombstone.authorization_epoch,
            &tombstone.signer_public_key,
        )?;
        verify_detached(&tombstone.signer_public_key, &bytes, &tombstone.signature)
    }

    fn put_encrypted<T: Serialize>(
        &self,
        definition: TableDefinition<&[u8], &[u8]>,
        key: &[u8; 32],
        value: &T,
    ) -> Result<()> {
        let bytes = postcard::to_allocvec(value)?;
        let (wrapped_key, encrypted) = seal_object(&self.keys, key, &bytes)?;
        let write = self.db.begin_write()?;
        {
            let mut table = write.open_table(definition)?;
            table.insert(key.as_slice(), encrypted.as_slice())?;
            let mut data_keys = write.open_table(DATA_KEYS)?;
            data_keys.insert(key.as_slice(), wrapped_key.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    fn get_encrypted<T: DeserializeOwned>(
        &self,
        definition: TableDefinition<&[u8], &[u8]>,
        key: &[u8; 32],
    ) -> Result<Option<T>> {
        let read = self.db.begin_read()?;
        let table = read.open_table(definition)?;
        let Some(value) = table.get(key.as_slice())? else {
            return Ok(None);
        };
        let data_keys = read.open_table(DATA_KEYS)?;
        let Some(wrapped_key) = data_keys.get(key.as_slice())? else {
            return Ok(None);
        };
        let bytes = open_object(&self.keys, key, wrapped_key.value(), value.value())?;
        Ok(Some(postcard::from_bytes(&bytes)?))
    }

    fn all_encrypted<T: DeserializeOwned>(
        &self,
        definition: TableDefinition<&[u8], &[u8]>,
    ) -> Result<Vec<T>> {
        let read = self.db.begin_read()?;
        let table = read.open_table(definition)?;
        let data_keys = read.open_table(DATA_KEYS)?;
        let mut values = Vec::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            let key: [u8; 32] = key.value().try_into().map_err(|_| HeartError::NotFound)?;
            let Some(wrapped_key) = data_keys.get(key.as_slice())? else {
                continue;
            };
            let bytes = open_object(&self.keys, &key, wrapped_key.value(), value.value())?;
            values.push(postcard::from_bytes(&bytes)?);
        }
        Ok(values)
    }

    fn crypto_shred(&self, target: &TombstoneTarget) -> Result<()> {
        let key = match target {
            TombstoneTarget::Event(id) => id.as_bytes(),
            TombstoneTarget::Snapshot(id) => id.as_bytes(),
            TombstoneTarget::Blob(id) => id.as_bytes(),
        };
        let write = self.db.begin_write()?;
        {
            let mut data_keys = write.open_table(DATA_KEYS)?;
            data_keys.remove(key.as_slice())?;
        }
        write.commit()?;
        if let TombstoneTarget::Blob(id) = target {
            let path = self.blob_path(*id);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

fn blob_directory(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".blobs");
    PathBuf::from(value)
}

fn tombstone_signing_bytes(tombstone: &Tombstone) -> Result<Vec<u8>> {
    Ok(postcard::to_allocvec(&(
        &tombstone.target,
        tombstone.device_id,
        tombstone.authorization_epoch,
        tombstone.device_sequence,
        tombstone.wall_millis,
        &tombstone.reason,
    ))?)
}

fn authorization_signing_bytes(authorization: &DeviceAuthorization) -> Result<Vec<u8>> {
    Ok(postcard::to_allocvec(&(
        authorization.device_id,
        authorization.signing_public_key,
        authorization.epoch,
        authorization.owner_public_key,
    ))?)
}

fn read_header(db: &Database) -> Result<StoreHeader> {
    let read = db.begin_read()?;
    let meta = read.open_table(META)?;
    let value = meta.get("header")?.ok_or(HeartError::UnlockFailed)?;
    Ok(serde_json::from_slice(value.value())?)
}

fn read_encrypted_meta<T: DeserializeOwned>(
    db: &Database,
    keys: &StoreKeys,
    key: &str,
) -> Result<T> {
    let read = db.begin_read()?;
    let meta = read.open_table(META)?;
    let value = meta.get(key)?.ok_or(HeartError::UnlockFailed)?;
    let aad = format!("meta:{key}");
    let plaintext = decrypt_record(keys, aad.as_bytes(), value.value())?;
    Ok(postcard::from_bytes(&plaintext)?)
}

fn write_encrypted_meta<T: Serialize>(
    db: &Database,
    keys: &StoreKeys,
    key: &str,
    value: &T,
) -> Result<()> {
    let bytes = postcard::to_allocvec(value)?;
    let aad = format!("meta:{key}");
    let encrypted = encrypt_record(keys, aad.as_bytes(), &bytes)?;
    let write = db.begin_write()?;
    {
        let mut meta = write.open_table(META)?;
        meta.insert(key, encrypted.as_slice())?;
    }
    write.commit()?;
    Ok(())
}

fn derive_owner_secret(keys: &StoreKeys) -> Result<[u8; 32]> {
    let hash = blake3::keyed_hash(&keys.root, b"spine-owner-signing-v1");
    Ok(*hash.as_bytes())
}

fn event_order_key(event: &SignedEvent) -> Vec<u8> {
    let mut key = Vec::with_capacity(8 + 4 + 32 + 8 + 32);
    key.extend_from_slice(&event.body.timestamp.wall_millis.to_be_bytes());
    key.extend_from_slice(&event.body.timestamp.counter.to_be_bytes());
    key.extend_from_slice(event.body.device_id.as_bytes());
    key.extend_from_slice(&event.body.device_sequence.to_be_bytes());
    key.extend_from_slice(event.id.as_bytes());
    key
}

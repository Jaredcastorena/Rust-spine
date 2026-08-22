use argon2::{Algorithm, Argon2, Params, Version};
use bip39::{Language, Mnemonic};
use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, KeyInit, Payload},
};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use hkdf::Hkdf;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

use crate::{DeviceId, EventBody, EventId, HeartError, Result, SignedEvent};

const ARGON_MEMORY_KIB: u32 = 64 * 1024;
const ARGON_ITERATIONS: u32 = 3;
const ARGON_LANES: u32 = 1;

#[derive(Clone, Debug, Zeroize)]
#[zeroize(drop)]
pub struct RecoveryPhrase(String);

impl RecoveryPhrase {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Zeroize)]
#[zeroize(drop)]
pub enum KeySource {
    Passphrase(String),
    RecoveryPhrase(String),
    RootKey([u8; 32]),
}

#[derive(Debug)]
pub struct CreatedKeys {
    pub recovery_phrase: RecoveryPhrase,
    pub root_key: Zeroizing<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WrappedRoot {
    pub salt: [u8; 16],
    pub nonce: [u8; 24],
    pub ciphertext: Vec<u8>,
    pub argon_memory_kib: u32,
    pub argon_iterations: u32,
    pub argon_lanes: u32,
}

#[derive(Debug, Zeroize)]
#[zeroize(drop)]
pub(crate) struct StoreKeys {
    pub root: [u8; 32],
    pub record: [u8; 32],
    pub object_id: [u8; 32],
    pub sync: [u8; 32],
}

#[derive(Debug, Zeroize)]
#[zeroize(drop)]
pub(crate) struct DeviceIdentity {
    signing_secret: [u8; 32],
}

impl DeviceIdentity {
    pub fn generate() -> Result<Self> {
        let mut signing_secret = [0_u8; 32];
        getrandom::fill(&mut signing_secret).map_err(|_| HeartError::Crypto)?;
        Ok(Self { signing_secret })
    }

    pub fn from_secret(signing_secret: [u8; 32]) -> Self {
        Self { signing_secret }
    }

    pub fn secret(&self) -> &[u8; 32] {
        &self.signing_secret
    }

    pub fn public(&self) -> [u8; 32] {
        SigningKey::from_bytes(&self.signing_secret)
            .verifying_key()
            .to_bytes()
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(*blake3::hash(&self.public()).as_bytes())
    }

    pub fn sign_event(&self, keys: &StoreKeys, body: EventBody) -> Result<SignedEvent> {
        let bytes = postcard::to_allocvec(&body)?;
        let id = EventId(*blake3::keyed_hash(&keys.object_id, &bytes).as_bytes());
        let signing = SigningKey::from_bytes(&self.signing_secret);
        let signature = signing.sign(&bytes).to_bytes().to_vec();
        Ok(SignedEvent {
            id,
            body,
            signer_public_key: signing.verifying_key().to_bytes(),
            signature,
        })
    }

    pub fn sign_bytes(&self, bytes: &[u8]) -> ([u8; 32], Vec<u8>) {
        let signing = SigningKey::from_bytes(&self.signing_secret);
        (
            signing.verifying_key().to_bytes(),
            signing.sign(bytes).to_bytes().to_vec(),
        )
    }
}

pub(crate) fn verify_detached(public: &[u8; 32], bytes: &[u8], signature: &[u8]) -> Result<()> {
    let public = VerifyingKey::from_bytes(public).map_err(|_| HeartError::InvalidSignature)?;
    let signature: [u8; 64] = signature
        .try_into()
        .map_err(|_| HeartError::InvalidSignature)?;
    public
        .verify(bytes, &ed25519_dalek::Signature::from_bytes(&signature))
        .map_err(|_| HeartError::InvalidSignature)
}

pub(crate) fn generate_root() -> Result<CreatedKeys> {
    let mut root = [0_u8; 32];
    getrandom::fill(&mut root).map_err(|_| HeartError::Crypto)?;
    let mnemonic = Mnemonic::from_entropy_in(Language::English, &root)
        .map_err(|_| HeartError::InvalidRecoveryPhrase)?;
    Ok(CreatedKeys {
        recovery_phrase: RecoveryPhrase(mnemonic.to_string()),
        root_key: Zeroizing::new(root),
    })
}

pub(crate) fn root_from_phrase(phrase: &str) -> Result<Zeroizing<[u8; 32]>> {
    let mnemonic = Mnemonic::parse_in_normalized(Language::English, phrase)
        .map_err(|_| HeartError::InvalidRecoveryPhrase)?;
    let entropy = mnemonic.to_entropy();
    let root: [u8; 32] = entropy
        .try_into()
        .map_err(|_| HeartError::InvalidRecoveryPhrase)?;
    Ok(Zeroizing::new(root))
}

pub(crate) fn derive_store_keys(root: &[u8; 32]) -> Result<StoreKeys> {
    let hk = Hkdf::<Sha256>::new(Some(b"spine-heart-v1"), root);
    let mut record = [0_u8; 32];
    let mut object_id = [0_u8; 32];
    let mut sync = [0_u8; 32];
    hk.expand(b"record-aead", &mut record)
        .map_err(|_| HeartError::Crypto)?;
    hk.expand(b"object-identifiers", &mut object_id)
        .map_err(|_| HeartError::Crypto)?;
    hk.expand(b"sync-envelope", &mut sync)
        .map_err(|_| HeartError::Crypto)?;
    Ok(StoreKeys {
        root: *root,
        record,
        object_id,
        sync,
    })
}

pub(crate) fn wrap_root(root: &[u8; 32], passphrase: &str) -> Result<WrappedRoot> {
    let mut salt = [0_u8; 16];
    let mut nonce = [0_u8; 24];
    getrandom::fill(&mut salt).map_err(|_| HeartError::Crypto)?;
    getrandom::fill(&mut nonce).map_err(|_| HeartError::Crypto)?;
    let wrapping_key = derive_passphrase_key(
        passphrase,
        &salt,
        ARGON_MEMORY_KIB,
        ARGON_ITERATIONS,
        ARGON_LANES,
    )?;
    let cipher = XChaCha20Poly1305::new((&*wrapping_key).into());
    let nonce_value = XNonce::from(nonce);
    let ciphertext = cipher
        .encrypt(
            &nonce_value,
            Payload {
                msg: root,
                aad: b"spine-wrapped-root-v1",
            },
        )
        .map_err(|_| HeartError::Crypto)?;
    Ok(WrappedRoot {
        salt,
        nonce,
        ciphertext,
        argon_memory_kib: ARGON_MEMORY_KIB,
        argon_iterations: ARGON_ITERATIONS,
        argon_lanes: ARGON_LANES,
    })
}

pub(crate) fn unwrap_root(wrapped: &WrappedRoot, passphrase: &str) -> Result<Zeroizing<[u8; 32]>> {
    let wrapping_key = derive_passphrase_key(
        passphrase,
        &wrapped.salt,
        wrapped.argon_memory_kib,
        wrapped.argon_iterations,
        wrapped.argon_lanes,
    )?;
    let cipher = XChaCha20Poly1305::new((&*wrapping_key).into());
    let nonce = XNonce::from(wrapped.nonce);
    let plaintext = cipher
        .decrypt(
            &nonce,
            Payload {
                msg: &wrapped.ciphertext,
                aad: b"spine-wrapped-root-v1",
            },
        )
        .map_err(|_| HeartError::UnlockFailed)?;
    let root: [u8; 32] = plaintext.try_into().map_err(|_| HeartError::UnlockFailed)?;
    Ok(Zeroizing::new(root))
}

fn derive_passphrase_key(
    passphrase: &str,
    salt: &[u8],
    memory_kib: u32,
    iterations: u32,
    lanes: u32,
) -> Result<Zeroizing<[u8; 32]>> {
    let params =
        Params::new(memory_kib, iterations, lanes, Some(32)).map_err(|_| HeartError::Crypto)?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = Zeroizing::new([0_u8; 32]);
    argon
        .hash_password_into(passphrase.as_bytes(), salt, &mut *key)
        .map_err(|_| HeartError::Crypto)?;
    Ok(key)
}

pub(crate) fn encrypt_record(keys: &StoreKeys, aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    encrypt_with_key(&keys.record, aad, plaintext)
}

pub(crate) fn decrypt_record(keys: &StoreKeys, aad: &[u8], envelope: &[u8]) -> Result<Vec<u8>> {
    decrypt_with_key(&keys.record, aad, envelope)
}

pub(crate) fn seal_object(
    keys: &StoreKeys,
    object_id: &[u8; 32],
    plaintext: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut data_key = Zeroizing::new([0_u8; 32]);
    getrandom::fill(&mut *data_key).map_err(|_| HeartError::Crypto)?;
    let ciphertext = encrypt_with_key(&data_key, object_id, plaintext)?;
    let mut key_aad = b"spine-object-key-v1:".to_vec();
    key_aad.extend_from_slice(object_id);
    let wrapped_key = encrypt_with_key(&keys.record, &key_aad, &*data_key)?;
    Ok((wrapped_key, ciphertext))
}

pub(crate) fn open_object(
    keys: &StoreKeys,
    object_id: &[u8; 32],
    wrapped_key: &[u8],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    let mut key_aad = b"spine-object-key-v1:".to_vec();
    key_aad.extend_from_slice(object_id);
    let data_key = Zeroizing::new(decrypt_with_key(&keys.record, &key_aad, wrapped_key)?);
    let data_key: &[u8; 32] = data_key
        .as_slice()
        .try_into()
        .map_err(|_| HeartError::Crypto)?;
    decrypt_with_key(data_key, object_id, ciphertext)
}

fn encrypt_with_key(key: &[u8; 32], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    let cipher = XChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0_u8; 24];
    getrandom::fill(&mut nonce_bytes).map_err(|_| HeartError::Crypto)?;
    let nonce = XNonce::from(nonce_bytes);
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| HeartError::Crypto)?;
    let mut envelope = Vec::with_capacity(24 + ciphertext.len());
    envelope.extend_from_slice(&nonce_bytes);
    envelope.extend_from_slice(&ciphertext);
    Ok(envelope)
}

fn decrypt_with_key(key: &[u8; 32], aad: &[u8], envelope: &[u8]) -> Result<Vec<u8>> {
    if envelope.len() < 24 {
        return Err(HeartError::Crypto);
    }
    let (nonce, ciphertext) = envelope.split_at(24);
    let nonce: [u8; 24] = nonce.try_into().map_err(|_| HeartError::Crypto)?;
    let nonce = XNonce::from(nonce);
    let cipher = XChaCha20Poly1305::new(key.into());
    cipher
        .decrypt(
            &nonce,
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| HeartError::Crypto)
}

pub(crate) fn verify_event(keys: &StoreKeys, event: &SignedEvent) -> Result<()> {
    let bytes = postcard::to_allocvec(&event.body)?;
    let expected = EventId(*blake3::keyed_hash(&keys.object_id, &bytes).as_bytes());
    if expected != event.id {
        return Err(HeartError::InvalidEventId);
    }
    let signer_device = DeviceId(*blake3::hash(&event.signer_public_key).as_bytes());
    if signer_device != event.body.device_id {
        return Err(HeartError::InvalidSignature);
    }
    let public = VerifyingKey::from_bytes(&event.signer_public_key)
        .map_err(|_| HeartError::InvalidSignature)?;
    let signature_bytes: [u8; 64] = event
        .signature
        .as_slice()
        .try_into()
        .map_err(|_| HeartError::InvalidSignature)?;
    let signature = ed25519_dalek::Signature::from_bytes(&signature_bytes);
    public
        .verify(&bytes, &signature)
        .map_err(|_| HeartError::InvalidSignature)
}

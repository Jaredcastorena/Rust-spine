use serde::{Deserialize, Serialize};

use crate::{HeartError, Result, vector};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema: u32,
    pub model_name: String,
    pub artifact_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub dimension: usize,
    pub normalized: bool,
    pub quantization: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn normalized(values: Vec<f32>, dimension: usize) -> Result<Self> {
        vector::validate_dimension(&values, dimension)?;
        let values = vector::unit(&values, 1e-12);
        if vector::norm(&values) < 1e-8 {
            return Err(HeartError::InvalidInput(
                "embedding must have nonzero magnitude".into(),
            ));
        }
        Ok(Self(values))
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }
}

pub trait SemanticEncoder: Send + Sync {
    fn manifest(&self) -> &ModelManifest;
    fn encode(&self, text: &str) -> Result<Embedding>;

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use spine_heart::{Embedding, HeartError, ModelManifest, Result, SemanticEncoder};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Clone, Debug)]
pub struct MiniLmAssets {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

impl MiniLmAssets {
    pub fn from_directory(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        Self {
            config: directory.join("config.json"),
            tokenizer: directory.join("tokenizer.json"),
            weights: directory.join("model.safetensors"),
        }
    }
}

pub struct MiniLmEncoder {
    manifest: ModelManifest,
    tokenizer: Tokenizer,
    model: BertModel,
    device: Device,
    maximum_tokens: usize,
}

impl MiniLmEncoder {
    pub fn load(assets: MiniLmAssets, maximum_tokens: usize) -> Result<Self> {
        if !(2..=512).contains(&maximum_tokens) {
            return Err(HeartError::InvalidInput(
                "MiniLM maximum tokens must be in 2..=512".into(),
            ));
        }
        let config_bytes = fs::read(&assets.config)?;
        let tokenizer_bytes = fs::read(&assets.tokenizer)?;
        let weight_bytes = fs::read(&assets.weights)?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        let dimension = config.hidden_size;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: maximum_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|error| HeartError::Model(error.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
        let device = Device::Cpu;
        let artifact_hash = *blake3::hash(&weight_bytes).as_bytes();
        let builder = VarBuilder::from_buffered_safetensors(weight_bytes, DType::F32, &device)
            .map_err(model_error)?;
        let model = BertModel::load(builder, &config).map_err(model_error)?;
        Ok(Self {
            manifest: ModelManifest {
                schema: 1,
                model_name: "sentence-transformers/all-MiniLM-L6-v2".into(),
                artifact_hash,
                tokenizer_hash: *blake3::hash(&tokenizer_bytes).as_bytes(),
                dimension,
                normalized: true,
                quantization: None,
            },
            tokenizer,
            model,
            device,
            maximum_tokens,
        })
    }

    pub fn maximum_tokens(&self) -> usize {
        self.maximum_tokens
    }

    fn encode_native_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        let token_ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect();
        let token_types: Vec<Vec<u32>> = encodings
            .iter()
            .map(|encoding| encoding.get_type_ids().to_vec())
            .collect();
        let masks: Vec<Vec<u32>> = encodings
            .iter()
            .map(|encoding| encoding.get_attention_mask().to_vec())
            .collect();
        let input_ids = Tensor::new(token_ids, &self.device).map_err(model_error)?;
        let token_type_ids = Tensor::new(token_types, &self.device).map_err(model_error)?;
        let attention_mask = Tensor::new(masks, &self.device).map_err(model_error)?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(model_error)?;
        let float_mask = attention_mask
            .to_dtype(DType::F32)
            .and_then(|mask| mask.unsqueeze(2))
            .map_err(model_error)?;
        let summed = hidden
            .broadcast_mul(&float_mask)
            .and_then(|weighted| weighted.sum(1))
            .map_err(model_error)?;
        let counts = float_mask
            .sum(1)
            .and_then(|counts| counts.clamp(1e-9, f64::INFINITY))
            .map_err(model_error)?;
        let pooled = summed.broadcast_div(&counts).map_err(model_error)?;
        pooled
            .to_vec2::<f32>()
            .map_err(model_error)?
            .into_iter()
            .map(|values| Embedding::normalized(values, self.manifest.dimension))
            .collect()
    }
}

impl SemanticEncoder for MiniLmEncoder {
    fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn encode(&self, text: &str) -> Result<Embedding> {
        self.encode_native_batch(&[text.to_owned()])?
            .into_iter()
            .next()
            .ok_or_else(|| HeartError::Model("MiniLM returned no embedding".into()))
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.encode_native_batch(texts)
    }
}

fn model_error(error: candle_core::Error) -> HeartError {
    HeartError::Model(error.to_string())
}

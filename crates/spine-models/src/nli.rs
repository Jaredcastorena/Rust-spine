use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use candle_transformers::models::bert::{BertModel, Config};
use spine_heart::{HeartError, ModelManifest, NliLabelOrder, NliModel, Result};
use tokenizers::{EncodeInput, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Clone, Debug)]
pub struct NliAssets {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

impl NliAssets {
    pub fn from_directory(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        Self {
            config: directory.join("config.json"),
            tokenizer: directory.join("tokenizer.json"),
            weights: directory.join("model.safetensors"),
        }
    }
}

pub struct MiniLmNli {
    manifest: ModelManifest,
    tokenizer: Tokenizer,
    model: BertModel,
    classifier_dense: Linear,
    classifier_output: Linear,
    device: Device,
}

impl MiniLmNli {
    pub fn load(assets: NliAssets) -> Result<Self> {
        let config_bytes = fs::read(&assets.config)?;
        let tokenizer_bytes = fs::read(&assets.tokenizer)?;
        let weight_bytes = fs::read(&assets.weights)?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        if config.model_type.as_deref() != Some("roberta") || config.hidden_size != 768 {
            return Err(HeartError::InvalidInput(
                "NLI assets are not the locked MiniLMv2 RoBERTa baseline".into(),
            ));
        }
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..TruncationParams::default()
            }))
            .map_err(|error| HeartError::Model(error.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 1,
            pad_token: "<pad>".into(),
            ..PaddingParams::default()
        }));
        let device = Device::Cpu;
        let artifact_hash = *blake3::hash(&weight_bytes).as_bytes();
        let builder = VarBuilder::from_buffered_safetensors(weight_bytes, DType::F32, &device)
            .map_err(model_error)?;
        // BertModel's transformer blocks are architecture-compatible with
        // RoBERTa. `forward_roberta` below supplies RoBERTa's +2 positional
        // offset by adding two masked padding positions.
        let model = BertModel::load(builder.clone(), &config).map_err(model_error)?;
        let classifier_dense = linear(
            config.hidden_size,
            config.hidden_size,
            builder.pp("classifier.dense"),
        )
        .map_err(model_error)?;
        let classifier_output = linear(config.hidden_size, 3, builder.pp("classifier.out_proj"))
            .map_err(model_error)?;
        Ok(Self {
            manifest: ModelManifest {
                schema: 1,
                model_name: "cross-encoder/nli-MiniLM2-L6-H768".into(),
                artifact_hash,
                tokenizer_hash: *blake3::hash(&tokenizer_bytes).as_bytes(),
                dimension: 3,
                normalized: false,
                quantization: None,
            },
            tokenizer,
            model,
            classifier_dense,
            classifier_output,
            device,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn predict(&self, pairs: &[(String, String)]) -> Result<Vec<[f32; 3]>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<EncodeInput<'_>> = pairs
            .iter()
            .map(|(first, second)| (first.as_str(), second.as_str()).into())
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|error| HeartError::Model(error.to_string()))?;
        let mut ids = Vec::with_capacity(encodings.len());
        let mut types = Vec::with_capacity(encodings.len());
        let mut masks = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            let mut row_ids = Vec::with_capacity(encoding.len() + 2);
            row_ids.extend_from_slice(&[1, 1]);
            row_ids.extend_from_slice(encoding.get_ids());
            ids.push(row_ids);
            let mut row_types = vec![0, 0];
            row_types.extend_from_slice(encoding.get_type_ids());
            types.push(row_types);
            let mut row_mask = vec![0, 0];
            row_mask.extend_from_slice(encoding.get_attention_mask());
            masks.push(row_mask);
        }
        let ids = Tensor::new(ids, &self.device).map_err(model_error)?;
        let types = Tensor::new(types, &self.device).map_err(model_error)?;
        let masks = Tensor::new(masks, &self.device).map_err(model_error)?;
        let hidden = self
            .model
            .forward(&ids, &types, Some(&masks))
            .map_err(model_error)?;
        let first_token = hidden
            .narrow(1, 2, 1)
            .and_then(|tensor| tensor.squeeze(1))
            .map_err(model_error)?;
        let logits = self
            .classifier_dense
            .forward(&first_token)
            .and_then(|tensor| tensor.tanh())
            .and_then(|tensor| self.classifier_output.forward(&tensor))
            .map_err(model_error)?;
        logits
            .to_vec2::<f32>()
            .map_err(model_error)?
            .into_iter()
            .map(|row| {
                row.try_into()
                    .map_err(|_| HeartError::Model("NLI classifier returned wrong width".into()))
            })
            .collect()
    }
}

impl NliModel for MiniLmNli {
    fn label_order(&self) -> NliLabelOrder {
        NliLabelOrder::CROSS_ENCODER_MINILM
    }

    fn predict_logits(&self, pairs: &[(String, String)]) -> Result<Vec<[f32; 3]>> {
        self.predict(pairs)
    }
}

fn model_error(error: candle_core::Error) -> HeartError {
    HeartError::Model(error.to_string())
}

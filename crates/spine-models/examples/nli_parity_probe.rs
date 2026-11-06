//! Offline diagnostic: evaluate premise/hypothesis pairs using the native NLI.
use std::fs;

use spine_heart::NliModel;
use spine_models::{MiniLmNli, NliAssets};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: nli_parity_probe MODEL_DIR PAIRS_JSON OUTPUT_JSON".into());
    }
    let pairs: Vec<(String, String)> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let model = MiniLmNli::load(NliAssets::from_directory(&args[1]))?;
    let logits = model.predict_logits(&pairs)?;
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(&serde_json::json!({
            "pairs": pairs, "logits": logits, "manifest": model.manifest(),
        }))?,
    )?;
    Ok(())
}

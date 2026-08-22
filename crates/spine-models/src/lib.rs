#![forbid(unsafe_code)]

mod minilm;
mod nli;

pub use minilm::{MiniLmAssets, MiniLmEncoder};
pub use nli::{MiniLmNli, NliAssets};

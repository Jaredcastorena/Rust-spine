# rust-spine

`rust-spine` is the standalone native Spine harness. It combines a long-running tool runtime with an encrypted portable heart containing canonical events, DCMDb memory, Thymos state, typed facts, learned risk, and triangle-compacted context.

The repository has no Python runtime dependency and does not modify the legacy Python DCMDb project.

## What is included

- `spine-heart`: encrypted event storage, identity, snapshots, sync, DCMDb, Thymos, facts, RiskField, and context triangles.
- `spine-models`: native Candle loaders for the MiniLM embedder and optional NLI verifier.
- `spine-runtime`: OpenAI-compatible provider, tool harness, live operator controls, checkpoints, and temporary subagents.
- `spine-cli`: the interactive partner, memory commands, action tools, ingestion, grounding, and terminal interface.
- `fuzz`: source-only libFuzzer targets. Generated corpora and artifacts are intentionally excluded.

Downloaded model weights, encrypted hearts, logs, build output, and API credentials are deliberately not part of the repository.

## Requirements

- Rust 1.98.0; the pinned toolchain file installs `rustfmt` and `clippy` through rustup.
- An OpenAI-compatible chat-completions server such as llama.cpp.
- A local `sentence-transformers/all-MiniLM-L6-v2` snapshot containing `config.json`, `tokenizer.json`, and `model.safetensors`.
- Optionally, a local `cross-encoder/nli-MiniLM2-L6-H768` snapshot with the same three asset files.

## Build and verify

```bash
cargo build --release --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --all -- --check
```

The resulting executable is `target/release/spine`.

## Start the partner

Keep secrets in the process environment rather than command history:

```bash
export SPINE_HEART_PASSPHRASE='your existing heart passphrase'
export SPINE_LLM_API_KEY='your server key'
export SPINE_MINILM_DIR='/path/to/all-MiniLM-L6-v2/snapshot'
export SPINE_NLI_DIR='/path/to/nli-MiniLM2-L6-H768/snapshot'

./target/release/spine chat "$HOME/.local/share/spine/default.spine" \
  --model-dir "$SPINE_MINILM_DIR" \
  --nli-model-dir "$SPINE_NLI_DIR" \
  --server-url http://127.0.0.1:9001 \
  --max-context-tokens 262144
```

Omit `--nli-model-dir` to run without host grounding. `--max-tool-rounds` is optional; tool rounds are unlimited by default.

On the first launch at a new heart path, Spine creates and initializes the encrypted heart and prints its recovery phrase. Store that phrase offline. Later sessions reopen the same heart when given the same path and passphrase. A wrong passphrase fails closed and never creates a replacement over an existing heart.

During a turn, type guidance and press Enter to queue it at the next completed-tool boundary. `/stop` requests a graceful checkpoint, `/interrupt` stops immediately, `/resume` continues a checkpoint, `/tasks` shows host work, and `/quit` exits at a safe boundary.

### Incognito test sessions

Use `--incognito-mode` when testing tools, prompts, or onboarding without touching the persistent heart:

```bash
./target/release/spine chat --incognito-mode \
  --model-dir "$SPINE_MINILM_DIR" \
  --nli-model-dir "$SPINE_NLI_DIR" \
  --server-url http://127.0.0.1:9001
```

`--test-mode` is a visible alias for the same behavior. No heart path or heart passphrase is required. Spine creates a private temporary directory and a randomly keyed heart, suppresses recovery material and snapshots, and removes the heart, encrypted blobs, and action audit when the process exits normally. Provider credentials and local model paths work exactly as they do in persistent mode.

### Web interface

Add `--web-server` to project the same partner session through the embedded browser UI:

```bash
./target/release/spine chat "$HOME/.local/share/spine/default.spine" \
  --web-server \
  --model-dir "$SPINE_MINILM_DIR" \
  --nli-model-dir "$SPINE_NLI_DIR" \
  --server-url http://127.0.0.1:9001
```

The default UI address is `127.0.0.1:9002`. Spine prints the complete access URL at startup; open that URL rather than the bare address because its fragment contains a random per-process browser token. The fragment is kept out of HTTP requests and moved into browser session storage. Every JSON API call must then present it through `X-Spine-Token`.

The browser and terminal share one harness, history, heart, task manager, and operator control plane. Live activity, tool calls/results, grounding, messages, guidance, graceful stop, interrupt, resume, and task inspection are available in the web view. The server accepts only loopback addresses or Tailscale's managed address range; wildcard and ordinary LAN binding are rejected.

The single-binary asset approach and visual direction are inspired by llama.cpp's MIT-licensed `llama-ui`, while Spine's compact HTML/CSS/JavaScript and host API are implemented specifically for the Spine harness.

## Memory boundary

This standalone baseline reads the new encrypted Rust heart only. The legacy Python DCMDb may run as a separate backup, but it is not a transparent fallback for `heart_recall`. Old memories must be imported or exposed through a deliberate dual-read bridge before they participate in the native partner's recall.

## Runtime files

For a heart at `default.spine`, runtime sidecars use the same base location:

- `default.spine`: encrypted redb heart.
- `default.spine.blobs/`: encrypted large-blob ciphertext.
- `default.action-audit.jsonl`: host action audit log.

These paths are ignored by Git. Never commit heart files, recovery phrases, passphrases, or provider keys.

See [PROJECT_STATE.md](PROJECT_STATE.md) for the accepted baseline and known integration boundary.

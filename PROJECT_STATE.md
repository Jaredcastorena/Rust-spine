# Project state

## Baseline

- Standalone Rust workspace with no Python runtime imports.
- One encrypted heart contains the signed canonical event log and its versioned cognitive projection.
- The projection unifies DCMDb, per-agent Thymos, typed facts, learned risk, and context triangles.
- Interactive chat registers the complete native cognition/action surface, persistent history, grounding repair, operator guidance, safe stopping, resumable checkpoints, host-tracked tasks, and curated temporary subagents.
- Tool rounds are unlimited by default and remain configurable with a positive ceiling.
- The llama.cpp/OpenAI-compatible provider uses retry/backoff, explicit context budgeting, TCP keepalive, fresh request connections, and causal transport diagnostics.
- Terminal activity updates in place rather than printing one line per tool event.
- `--incognito-mode` (alias `--test-mode`) runs against a randomly keyed heart in a private temporary directory and does not require or modify a persistent heart.

## Security invariants

- The heart path identifies persistent state; a passphrase only unlocks it.
- Existing hearts fail closed on a wrong passphrase.
- Models receive an already-unlocked host tool handle and never receive the heart passphrase.
- Destructive actions remain host-gated.
- Heart data, blob sidecars, audit logs, model weights, credentials, and recovery material are not tracked by Git.

## Known boundary

Legacy Python DCMDb/Thymos state is not queried by the native runtime. Running the legacy process beside this harness does not create shared recall. Continuity requires either a one-time verified import or an explicit dual-read adapter.

## Accepted verification

- `cargo test --workspace --locked`: 53 tests.
- Strict workspace Clippy with all targets and features.
- Formatting check across the workspace.
- Optimized release build.

Update this file when an architectural boundary or accepted verification baseline changes.

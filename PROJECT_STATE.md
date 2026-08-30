# Project state

## Baseline

- Standalone Rust workspace with no Python runtime imports.
- One encrypted heart contains the signed canonical event log and its versioned cognitive projection.
- The projection unifies DCMDb, per-agent Thymos, typed facts, learned risk, and context triangles.
- Interactive chat registers the complete native cognition/action surface, persistent history, grounding repair, operator guidance, safe stopping, resumable checkpoints, host-tracked tasks, and curated temporary subagents.
- Multi-step work uses a host-owned plan cursor. Tool-free promises trigger another turn, steps advance only after evidence, and open plans survive graceful-stop checkpoints.
- Native OpenAI tool calls and the reference fenced-JSON/Laguna fallback forms share one bounded parser and execution path.
- Tool rounds are unlimited by default and remain configurable with a positive ceiling.
- The llama.cpp/OpenAI-compatible provider uses retry/backoff, explicit context budgeting, TCP keepalive, fresh request connections, and causal transport diagnostics.
- Provider startup combines `/props` and `/v1/models` discovery to adopt the
  active model ID and allocated context window; explicit operator settings take
  precedence, and tool schemas are included in the context budget. Compaction
  pins the active user instruction throughout tool loops.
- Terminal activity updates in place rather than printing one line per tool event.
- A new or empty heart starts with an adaptive model-led first conversation. The
  host enforces 5–10 one-at-a-time questions while the model chooses their content
  from prior answers; encrypted turns seed DCMDb and a distilled interaction
  profile supplies revisable behavioral defaults on later runs.
- `--incognito-mode` (alias `--test-mode`) runs against a randomly keyed heart in a private temporary directory and does not require or modify a persistent heart.
- `--web` serves an embedded single-binary UI on `127.0.0.1:8088` by default. Browser messages and controls enter the same harness loop as the terminal; no parallel agent session is created.
- `--llama-server-bin` plus `--llama-model` optionally manages the local llama.cpp server in the same process tree; using an existing OpenAI-compatible endpoint remains supported.
- Persistent chat has a platform-style default heart path, and MiniLM assets resolve from `SPINE_MINILM_DIR` or the local Hugging Face cache; explicit paths still override both.
- Web APIs require a random per-process fragment token. Non-loopback binding is rejected unless the operator explicitly supplies `--allow-remote-web`.
- Version tags build checksum-paired Linux, macOS, and Windows release archives; each archive contains one executable plus licenses and attribution.
- A two-stage OCI build runs the single binary as an unprivileged user, keeps
  hearts, workspaces, and model weights in separate mounts outside image layers,
  handles `SIGINT` for clean shutdown, and health-checks the actual PID 1 Spine
  process.

## Security invariants

- The heart path identifies persistent state; a passphrase only unlocks it.
- Existing hearts fail closed on a wrong passphrase.
- Models receive an already-unlocked host tool handle and never receive the heart passphrase.
- First-conversation prompts prohibit requests for secrets or identifying
  demographics. The resulting encrypted profile is bounded, treated as data,
  and cannot override current user instructions or host safety policy.
- Destructive actions remain host-gated.
- A failed grounding verifier is non-fatal; if a completed repair remains unsupported, the visible and persisted answer receives an explicit terminal caveat.
- Heart data, blob sidecars, audit logs, model weights, credentials, and recovery material are not tracked by Git.
- Runtime defaults and documentation contain no developer-specific hostnames, paths, addresses, or credentials.

## Known boundary

Legacy Python DCMDb/Thymos state is not queried by the native runtime. Running the legacy process beside this harness does not create shared recall. Continuity requires either a one-time verified import or an explicit dual-read adapter.

## Accepted verification

- `cargo test --workspace --locked`: 104 tests.
- Strict workspace Clippy with all targets and features.
- Formatting checks across the workspace and standalone fuzz package.
- Locked fuzz-target compilation.
- Locked dependency audit with no known vulnerabilities. The main graph retains
  one allowed unmaintained warning for `paste`, inherited through the current
  Candle/tokenizers stack; the fuzz graph is clean.
- Optimized release build, executable privacy scan, encrypted-heart lifecycle
  smoke test, and checksum/unpack/version smoke test of the Linux release archive.
- The release executable completed both terminal and authenticated browser chat
  turns against a local OpenAI-compatible test endpoint while using the genuine,
  checksum-verified MiniLM snapshot. Incognito hearts were removed on exit and
  the browser state exposed no local heart directory.
- An isolated `cargo install --locked --path crates/spine-cli` installs exactly
  one executable.
- A deterministic isolated smoke suite executes all 25 possible registered
  tools (24 in the default configuration), including task cancellation,
  browser history/search, document deduplication, encrypted cognition, and
  temporary sub-agent lifecycle. The harness also blocks a real destructive
  shell request by default.
- The 22.1 MB unprivileged runtime image completed the adaptive first
  conversation and live ordered filesystem/shell/read workflows against an
  external OpenAI-compatible endpoint, discovered changing 4,096- and
  262,144-token runtime allocations, reported healthy as PID 1, ran tools as UID
  10001, and preserved read-only model mounts. The required external MiniLM
  snapshot measured 91.6 MB; build-only storage is disclosed in the README.
- The generated third-party license bundle covers the normal dependency graph
  for every release target and is freshness-checked in CI and release preflight.
- CI and release workflows pass current `actionlint` validation.

Update this file when an architectural boundary or accepted verification baseline changes.

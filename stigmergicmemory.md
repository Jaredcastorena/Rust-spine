# Rust Spine working memory

## Current goal — 2026-10-15
- Change only chat startup: prompt for an OpenAI-compatible endpoint and port by
  default; require explicit --llamacpp to launch a managed local server.
- Implement and test on the clean PR #16 checkout, branch fix/python-parity.
  Its starting HEAD and fetched remote HEAD are both 3931bc5.
- Preserve unrelated uncommitted Codex/provider changes in the parent checkout.
- Explicit --server-url / SPINE_LLM_URL bypass prompts; noninteractive startup
  must require an explicit URL rather than consuming piped chat input.
- Prompt defaults: http://127.0.0.1/v1/chat/completions, port 8080. Normalize the
  standard full chat endpoint to the provider base URL; reject legacy text-only
  /v1/completions rather than silently sending an incompatible request.
- Implementation and local verification complete; delivery uses existing PR #16.
- Verification: 217 workspace tests pass, strict all-target/all-feature Clippy,
  formatting, diff/release hygiene, optimized build and real-terminal startup
  smoke pass. No model assets or production service were needed for this change.
- Independent review found two issues, both fixed: empty port delimiters must not
  imply port80, and full chat endpoints must normalize to the server prefix so
  root /health and /props discovery still works. Models-only and root-health-only
  mock-server regressions pass. Final reviewer reports no remaining blockers.
- Redirected stdin, explicit flag/environment URLs, EOF, invalid URLs/ports and
  llama environment paths without the opt-in flag have regression coverage.
- New-commit remote CI is not yet verified; use PR16 checks for its current state.

## Locked contracts
- Python-free production; preserve DCMDb/Thymos equations and signed event truth.
- Unlimited tool rounds remain default; explicit saved checkpoint policy survives.
- Preserve fact schema4 cleanup, learned state, source authority and atomic recovery.
- Triangles use real existing coordinates; no synthetic scaffold or periodic fold.
- Legacy Python import, device-revocation redesign and model selection are out of scope.

## Verified historical baseline
- September 8 correction: 206 Rust tests and 149 isolated Python tests passed.
- Independent reviews and native-model/release checks recorded in the archive.
- These historical results do not substitute for verification of today's change.

## Immutable archive
- Path: memory/archive/ccf406391500972157b000600dafcbe0b7d26f579c9f5a238d089fe2101e44c4.md
- SHA-256: ccf406391500972157b000600dafcbe0b7d26f579c9f5a238d089fe2101e44c4
- 97 lines; byte-for-byte copy and hash verified before shrinking this file.
- 1-based topic ranges: purpose 1–8; architecture 9–26; production workflow 27–35;
  parity integration/acceptance 36–73; September 8 review fixes 74–97.
- Never edit the archive or reuse its line addresses for changed content.

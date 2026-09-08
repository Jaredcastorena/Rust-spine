# Rust Spine parity ledger

## Purpose

Keep the implementation contracts, review findings, and verification evidence
durable as Rust becomes the base implementation. Read this before starting a
parity lane; update it after accepted milestones and when the next action changes.

## Architecture contracts

- Production is Python-free. The legacy Python implementation is a behavior oracle.
- Preserve the tested DCMDb and Thymos equations.
- Typed facts, supersession, recency, and aggregation derive from signed heart events.
- Preserve exact NLI evidence and the contradiction/entailment/neutral label order.
- Tool rounds are unlimited by default; positive operator ceilings remain optional.
- Guidance enters after a completed tool boundary. Graceful stop leaves a validated
  resumable checkpoint; immediate interrupt does not imply successful completion.
- Triangles contain addresses, use existing coherent DCMDb coordinates, and retain
  standalone branches when no valid apex exists. No synthetic scaffold or periodic
  three-to-one fold is introduced. Rehydration remains externally budgeted.
- Canonical events are durable truth. Recovery must preserve valid learned state;
  suffix catch-up must reject events that reorder an already projected prefix.
- Historical source provenance must never be rebound through today's filesystem.
- Legacy state import, new device-revocation design, and smaller model selection
  are separate projects from Python runtime parity.

## Production workflow

1. Reproduce each gap against the Python oracle and record the exact contract.
2. Implement in an isolated branch with deterministic regression coverage.
3. Have a second reviewer inspect behavior, error paths, and the tests.
4. Resolve findings and integrate focused commits into the parity branch.
5. Run the combined workspace tests, strict Clippy, formatting, release build,
   fuzz compilation, release hygiene, and license checks before PR acceptance.

## 2026-09-07 integration in progress

Provider wire compatibility, multiline terminal input, restart checkpoint
discovery, and source-aware document ingestion are integrated. Typed-fact and
host-modulation lanes are undergoing independent review. Resilience fixes include
subsystem breakers, browser command completion, suffix projection catch-up, and
protection of context coordinates during pruning.

Integration must reconcile ambiguous checkpoint-consumption writes using the
exact checkpoint event ID in canonical records. Unavailable risk must select a
conservative policy and remain visibly unavailable. Saved unlimited tool policy
must remain unlimited after restart, regardless of a new harness default.

The integrated branch is not yet accepted as parity-complete; combined gates and
final review remain pending.

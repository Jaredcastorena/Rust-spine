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

## 2026-09-07 parity integration

Current work is PR #16, `fix/python-parity`, based on current main `d8299d1`.
Provider/input/checkpoints, typed facts/aggregation, source-aware documents,
resilience, host modulation, introspection, and web fallback are integrated and
reviewed. `PARITY.md` maps each behavior to regression evidence.

Important accepted review findings:

- Exact canonical consume markers reconcile ambiguous writes before retry or
  resume; unknown markers fail closed. Saved unlimited tool policy stays
  unlimited across restart. Grounding repair preserves cumulative budgets and
  evidence, and cannot continue a gracefully stopped run.
- Fact schema 3 backfills user facts and removes document-source contamination
  while preserving non-fact cognitive state. Both chat and one-shot harness
  startup perform suffix recovery followed by fact upgrade.
- Released risk models used raw affect and count/empty features. Atomic migration
  preserves their weights in explicit compatibility segments and adds normalized
  affect plus six oracle inputs with zero initial weights. Fresh hearts use only
  the Python layout; unsupported layouts fail visibly.
- Unavailable risk is labeled unknown and selects conservative policy. Turn
  introspection is host-owned and bounded; resume clears unrelated turn metadata.
- Diagnostic history is a separate encrypted projection with known observations,
  bounded 100-sample retention and last-ten means. Missing legacy samples stay
  unknown; tensor replacement must break diagnostic continuity explicitly.

The completed implementation passes 201 combined Rust tests and the 149-test
isolated Python oracle. Actual native model tests, strict Clippy, formatting,
optimized build, fuzz compilation, license freshness, hygiene and dependency
audits pass. The release CLI completes two native tool rounds against a local
test endpoint using genuine MiniLM weights and cleans up its incognito heart.
Diagnostic suffix recovery is atomic, preserves learned state and context links,
and adds each known observation once across failure, retry and reopen.

Use PR #16's checks for current remote CI status. Review and merge remain normal
repository acceptance steps; do not silently change the locked architecture or
reinterpret missing historical diagnostics as known data.

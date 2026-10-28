# Changelog

## 0.3.2 — 2026-10-28

- Bound safe-stop summaries and streamed-response memory, recover failed-child
  results immediately, reject malformed recovery records, and prevent consumed
  checkpoints from resurfacing as cached obligations. Extend whole-workspace
  PR tests to Windows, both macOS architectures and ARM Linux. Release packages
  must pass the complete workspace tests on every target before publication.

- Restore optional DCMDb source/PMI/merge policies, presets, tuning and inspection;
  correct concentration and split-promotion differences found with Python fixtures.
- Restore raw Thymos queries, affinities, learning-rate overrides and presets;
  correct tied-channel selection and compare every activation/valence mode.
- Capture signed memory affinity, rank context by query and retained goals,
  and give child agents shared recall, meaningful check-ins and durable results.
- Preserve interrupted tool work in encrypted local journals; never automatically
  replay uncertain side effects. Retain existing desktop tool and action limits.
- Stream model output, separate inline thoughts, support advertised text-only
  tool interfaces and keep cancellation scoped to its request.
- Automatically install the pinned NLI answer checker with verified downloads;
  preserve local paths, offline operation and the explicit disable option.
- Remove unused direct dependencies and refresh lockfiles/license attribution.
  Existing heart layouts and older executable interoperability are preserved.

## 0.3.1 — 2026-10-24

- Includes the automatic setup and maintenance changes below.
- Windows release binaries statically link the C runtime, removing the separate
  Visual C++ Redistributable installation step. Release checks reject external
  Visual C++ runtime DLL dependencies.

## 0.3.0 — 2026-10-24 (validation tag; downloads not published)

- Run `spine` to start chat. First-run setup asks for one model server URL,
  checks readiness and API access, downloads verified MiniLM assets when needed,
  and remembers successful nonsecret connection and launch preferences.
- Saved profiles retain heart/model locations, workspace, interface selection,
  and optional NLI/onboarding preferences. Failed setup preserves the last
  working profile. Secrets and elevated permissions are never saved.
- File reads support Unicode character offsets, bounded pages and `next_offset`,
  preserving continuation information within the harness's output budget.
- Updated stable Rust dependencies, Rust 1.99.1, pinned GitHub Actions, and
  pinned Rust/Alpine container images. Container certificate roots support the
  updated TLS client and native memory-model downloads.
- Fixed the inherited ZIP extraction advisories with ZIP 9.1.0 and a small,
  provenance-checked Candle compatibility backport. Both dependency audits report
  zero known vulnerabilities. Candle's upstream `paste` dependency still carries
  an informational unmaintained warning.
- Existing encrypted heart data remains compatible: migration checks read and
  extend a heart created by v0.2.0 and reopen it with the old executable.

See [README.md](README.md) for setup, profile overrides and offline use, and
[vendor/README.md](vendor/README.md) for the temporary dependency backport.

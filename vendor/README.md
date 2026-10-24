# Temporary Candle ZIP compatibility backport

`candle-core/` is the published **0.11.0** crate, with its license retained.
The registry archive's SHA-256 and every included file's expected hash are in
`candle-core-source.json`. The upstream crate's lockfile and Cargo cache metadata
are omitted; Spine's root lockfile controls the dependency graph.

Candle 0.11.0 requires ZIP 8.6.0, which is affected by
RUSTSEC-2026-0367, RUSTSEC-2026-0368, and RUSTSEC-2026-0369. This copy uses ZIP 9.1.0
and adapts the three filename reads to ZIP 9's fallible `Cow<str>` API. The API
changes are backported from [Candle PR 4041](https://github.com/huggingface/candle/pull/4041),
revision `70f72e9cfdbfc570846ca363ead2135394af3e9a`.

Only `Cargo.toml`, `src/npy.rs`, `src/pickle.rs`, and `src/cpu/erf.rs` differ from the published
source. The last file uses `f64::INFINITY`/`NEG_INFINITY` instead of their
deprecated `std::f64` spellings. No tensor operations or model equations are changed. Regression tests
in `crates/spine-models/tests/zip_compat.rs` exercise UTF-8 and CP437 names and
malformed archives. The ordinary MiniLM/NLI and heart tests remain required.

Run `python3 scripts/check-vendored-candle.py` to verify the complete local file
inventory. The release hygiene check runs it in CI as well. No RustSec advisory
is ignored. Once a published Candle release supports ZIP 9, remove the root
`[patch.crates-io]` entry, this copy, and its provenance/checker, then refresh
lockfiles and licenses and rerun model and compatibility checks.

# Desktop feature preservation and Python behavior audit

Scope: the released desktop workspace, starting from v0.3.1 and its test-only
follow-up. Android and the separate Codex-provider work are outside this change.
The reference is the Python source supplied with the project, including its
DCMDb and Thymos packages. Production Python databases are never used as fixtures.

## Acceptance result

The desktop audit restores the missing cognitive controls and runtime behaviors
listed below. The complete Rust suite passes 253 tests, compared with 231 at the
starting release. The isolated Python suite passes 257 tests. These establish
specific behavioral evidence, not a claim of identical model-generated answers
or large-corpus performance. Source hashes and synthetic inputs are recorded in
the checked-in fixtures. No existing command, registered tool, or saved-heart
field was removed. Android and Codex-provider branches were left unchanged.

## Cleanup rules and baseline

- Inspect all four workspace manifests, native Rust modules, tests, embedded web
  assets, release scripts, and their entry points. Preserve public APIs, commands,
  optional tools, and serialized fields even when they lack an internal caller.
- Removed unused `uuid` and `x25519-dalek` direct dependencies and the runtime
  crate's unused `tempfile` test dependency. Hash-derived identifiers and the
  existing signing/encryption/sync implementation remain in place.
- Removed a stale dead-code suppression on the LongMemEval question ID; that
  field remains required and still identifies malformed dataset records.
- Before changes: 231 Rust workspace tests and strict Clippy passed. An isolated
  source-only Python run passed 257 tests, including DCMDb and Thymos. The first
  Rust run could not bind local test servers; the first Python run lacked an NLI
  asset path. Both environment failures were retained and corrected for reruns.

## DCMDb

| Python behavior | Desktop implementation and evidence |
| --- | --- |
| Spherical observations, trust-region updates, sufficient statistics, decay and confidence | `dcmdb.rs`; multi-observation Python fixtures compare node state, source histograms, graph weights and recall. |
| Concentration near a unit resultant | Fixed f32 rounding of the `1 - 1e-9` regularizer by evaluating the denominator in f64. |
| Tension detection, counter promotion, Bayesian split and clearing | Fixed counter promotion to use the tension center; recompute parent concentration/confidence after subtracting split evidence. Both paths have Python-generated regression fixtures. |
| Coupled, decoupled and hybrid coactivation | `DcmdbPolicy::pmi_mode`, checked against Python. |
| Positive, normalized and shifted PMI; G2 and chi-square weighting | Explicit policy controls, 27 Python combinations. |
| Source-aware candidate selection, PageRank seeding and query scores | Policy source weights and `source_probability`; 81 Python query comparisons. |
| Secondary-neighbor update limit | Optional policy cap, including zero and unlimited. |
| Semantic, graph, freshness, confidence, tension and MMR ranking | Retained; MMR overfetch now uses saturating arithmetic for extreme requested limits. |
| Adaptive merge-angle gate and decayed sample counts | Optional policies restored; eight Python merge cases cover adaptive gates, margins and decayed counts. |
| Merge likelihood, concentration/source guards, hierarchy, pruning and dreams | Existing maintenance tests; deterministic dream walks retained for signed-state reproducibility. |
| Hashing/dense/book presets, sample tuning, weight calibration | Native config constructors and explicit policy; Python fixtures cover small-sample tuning, scoring calibration and diagnostics. Large samples are selected deterministically. |
| Node, children, subtree, level and distribution inspection | Native inspection methods; runtime retrieval continues to use bounded expansion. |
| ANN acceleration | Exact current-centroid search is retained. Equivalent retrieval purpose, without Python's optional approximate-index performance. No large-corpus performance-equivalence claim. |
| JSON state and text index persistence | Replaced by encrypted canonical events, projections, snapshots and sync. Existing Python files are not automatically imported or overwritten. |

Optional algorithm policies are passed explicitly to update/query/maintenance;
they do not add fields to the released binary projection format. A library caller
using custom policies must persist the policy with its own operation log and
supply it consistently. Desktop's existing defaults remain unchanged.

The Python reference caches its nearest-neighbor index until the number of nodes
changes, leaving old centroids after ordinary updates. The fixture generator
explicitly refreshes that index before each update. This compares the Python
equations against current-centroid lookup without reproducing the stale-index
bug. This adjustment is recorded in the generated fixture itself.

## Thymos

| Python behavior | Desktop implementation and evidence |
| --- | --- |
| Tensor responses, none/tanh/ReLU/softmax, labels, valence and arousal | 24 generated configurations and 96 sequential observations. |
| Dominant channel | Fixed equal-response ordering to choose the first channel, matching NumPy; includes all-zero inputs. |
| Optional input normalization | Default normalized query retained; explicit raw-magnitude query restored. |
| All three prediction-error modes and three-factor learning | Cross-language fixtures cover eligibility, tensor and channel mass after updates, including row normalization. |
| Absolute learning-rate override | Restored alongside existing multipliers, including a nonzero override when the configured rate is zero. |
| Trajectory, surprise, heading and prediction | Generated sequential trajectories compare Python outputs; read-only trajectory-position access restored. |
| MiniLM, larger-embedding and minimal presets; sorted affinities | Restored native constructors and channel inspection. |
| Feeling history and last-ten trends | Existing encrypted per-agent diagnostic history; projection roundtrip and history tests retained. |
| NPZ/config persistence | Replaced by the versioned encrypted heart; no tensor/state fields removed. |

Fixtures are synthetic JSON produced by `scripts/generate-cognitive-oracles.py`
and `scripts/generate-dcmdb-oracles.py`. They record source hashes and require no
Python runtime during Rust tests. Numerical tolerances account for Rust's f32
state; exact IDs, rank ordering and discrete decisions are checked separately.

## Whole-application preservation checklist

| Area | Baseline / remaining acceptance |
| --- | --- |
| Automatic setup, model downloads, connection validation, saved launch settings | Endpoint/setup regressions plus an optimized-executable clean-install smoke: automatic MiniLM/NLI downloads, checksums, connection validation, saved settings, reuse, overrides and failure preservation. |
| Encrypted heart lifecycle, recovery, snapshots, sync, cold blobs and redaction | Store/projection/blob/property tests pass. A v0.2.0 heart was read and extended, then reopened by v0.2.0, including after new journal/context writes. |
| Facts, temporal supersession, aggregation, provenance and document ingestion | Existing fact oracles and tool smoke tests retained. |
| Feel/recall/reflect/risk modulation | Affinity is captured in signed event metadata; recall averages canonical affinities and applies the Python risk-weighted bias. Neutral/legacy vectors and concurrent projection updates have regressions. |
| NLI grounding and answer repair | Enabled by default with pinned, verified NLI installation. Explicit local paths, `--no-download`, and `--no-nli` remain supported. Genuine model tests distinguish entailment and contradiction. |
| Context compression and rehydration | Verified real triangles retained. Query/goal/tension/diversity rank roots; a regression reaches older relevant context. Goals, open obligations and evidence survive encrypted cold reopen. |
| Tool parsing, provider requests, discovery, reasoning and cancellation | Inline thoughts are separated before parsing; advertised non-native endpoints receive tool schemas in text. SSE assembly, rejected-stream fallback, distinct native calls and scoped cancellation have local-server regressions. |
| Filesystem, shell, web, task tracking and safety controls | Existing registered-tool and local-HTTP regressions retained. |
| Temporary subagents | Children receive refreshed shared recall, meaningful `check_in`/final reports, unique identities and durable transcripts. Final check-in avoids another model request. Delegated task text cannot become personal user facts; desktop round limits remain enforced. |
| Plans, guidance, stop/restart and checkpoint consumption | Existing controls pass. Encrypted device-local journals save intent before execution and results afterward. Provider failure remains an error for circuit handling. Recovery is idempotent and marks uncertain actions without executing them; failed journal writes stop the next side effect. |
| Terminal and authenticated embedded browser UI | Terminal setup smoke and authenticated browser smoke pass with genuine embeddings. Browser receives live text/reasoning and clears transient output at completion; unauthorized API access is rejected. |
| Evaluation/import tooling, offline models, container and release packaging | Commands and packaging retained; strict Clippy, format, locked fuzz compilation, optimized build, license freshness, privacy scan and both dependency audits pass. The inherited unmaintained `paste` warning remains; no vulnerability is suppressed. |

Legacy demonstration scripts, plotted dashboards, Python-specific module names,
and alternate inference-library implementations are not silently treated as
desktop runtime features. The native tools cover their operational purposes; Python plotting/CLI formatting
and optional approximate-index acceleration are not reproduced as separate
interfaces. Channel norms, mass, trajectory and feeling history remain available
through native diagnostics. No whole-application quality-equivalence claim
follows from numerical unit tests alone.

## Reproducing acceptance

Run the locked workspace tests and strict all-target/all-feature Clippy, then
`cargo check --manifest-path fuzz/Cargo.toml --locked`. Run the existing release
hygiene and third-party-license scripts, plus audits of both lockfiles. Set
`SPINE_MINILM_DIR` and `SPINE_NLI_DIR` to real local snapshots to exercise the two
model integration tests; without those variables these optional tests skip model
execution. The numerical fixtures need no model download or Python installation.

The NLI installer pins `cross-encoder/nli-MiniLM2-L6-H768` at
`b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`; all three installed files were independently
verified against the Hub. Its approximately 330 MB download is additional to
MiniLM's 91 MB. No new release tag is created by this audit.

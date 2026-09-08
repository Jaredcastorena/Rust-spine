# Python-to-Rust behavior acceptance

Rust is the base implementation. Python Spine is the behavioral oracle, not a
runtime dependency or a shared mutable database. This checklist distinguishes
implemented behavior from release acceptance; passing a lane alone does not
accept the combined branch.

| Behavior | Rust regression evidence |
| --- | --- |
| Typed user facts, dated paired turns, supersession and recency | `fact_parity`, `facts_and_verifier`, projection roundtrip |
| Natural sum/count/difference/extrema with exact evidence | Fact oracle families, ranking/tie-order and aggregation tests |
| Historical fact upgrade without resetting learned cognition | Atomic fact-upgrade and concurrent-change tests |
| Document identity, whitespace, retries, source authority | Document ingestion tests and real-heart tool smoke suite |
| OpenAI tool history and optional reasoning effort | Captured provider-request tests |
| Unlimited rounds, guidance boundaries, safe stop and resume | Harness control and restart checkpoint tests |
| Grounding repair preserves spent budgets and graceful stop | Repair continuation and checkpoint counter regressions |
| Surprise, tension, risk, temperature, action and recall controls | Modulation oracle and bounded expansion tests |
| Reflection equations and deterministic signed replay | Fixed-tensor cognitive oracles and encrypted reopen/rebuild |
| Six retrieval features and normalized affect | Risk oracle and learned-state-preserving migration tests |
| Feel and memory-distribution introspection | Host metadata isolation and real-heart introspection tests |
| Degraded operation and exact durable checkpoint consumption | Circuit-breaker and actual partial-failure recovery tests |
| Source-safe context coordinates and bounded rehydration | Protected pruning, triangle and property tests |
| Filesystem, shell, browser, tasks and temporary agents | Full registered-tool smoke tests and local HTTP regressions |
| Ordered search-engine fallback | Local failure/success/total-failure and exclusive-override tests |
| Native embedding and three-way NLI model contract | Actual local MiniLM and NLI model tests |

## Compatibility decisions

- Released four-feature risk models used count/empty inputs and raw affect.
  Their weights are retained in explicit compatibility segments; new normalized
  affect and six-feature coordinates start at zero. Fresh hearts use only the
  Python feature layout. No learned weights are silently reinterpreted or reset.
- Imported documents remain canonical searchable evidence, not the user's own
  personal facts. Fact schema 3 removes older document-derived contamination
  while retaining real user observations and all non-fact cognitive state.
- Unsupported or damaged projection layouts fail visibly. A fact upgrade cannot
  invent an original memory coordinate if all provenance has already been pruned.
- Restart uses exact checkpoint event IDs. An uncertain write is reconciled
  against verified canonical markers before deciding whether to restore or resume.

## Intentional differences

The architecture decisions in `stigmergicmemory.md` take precedence over a literal
port: no synthetic triangle scaffold or periodic three-to-one folding, no Python
production dependency, and no legacy Python state import requirement. New device
revocation and smaller-model selection remain separate projects.

## Final acceptance

Final acceptance is pending persisted diagnostic-history integration and the
complete checks on the final PR head. Required checks are workspace tests, actual
native-model tests, strict Clippy, formatting, optimized build, fuzz compilation,
dependency audit, license freshness, release hygiene, and independent review.

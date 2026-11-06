# Answer grounding

Spine checks a completed answer against the evidence available in its current
turn. The same check runs after a grounding repair and when a resumed run
finishes. `--no-nli` explicitly disables answer checking.

## Evidence and coverage

Evidence includes the current user request and later user guidance, eligible
recalled memory records (including canonical records freshly rehydrated into
triangle context), and successful tool results from the current turn.
Recalled records retain their event IDs and provenance. Assistant-authored
memories, model-requested saves, failed or ambiguous tool results, system
instructions, retained working summaries, and host repair feedback cannot corroborate the answer that
produced them. Earlier conversation messages are not automatically treated as
current evidence.

The harness retains the turn's messages independently of the provider's compacted
request. Additional recall during repair is therefore available for the second
check. Older checkpoints do not reliably identify their original evidence
boundary; resumed work can retrieve those sources again.

Every assessed statement receives an explicit disposition:

| Disposition | Meaning |
| --- | --- |
| Supported | Available evidence supports this statement. |
| Contradicted | Relevant evidence is incompatible with the statement. |
| Unverified | Available evidence or the checking budget does not establish the statement. |
| Conflicting | Relevant evidence supports incompatible statements. |
| Non-factual | The segment is a recognized question, instruction, acknowledgment, hypothetical, or other non-assertive output. |

Unrecognized factual wording remains in the checking path. Empty extraction is
not used as evidence that an answer contains no facts. Mixed answers are checked
by their constituent statements; one supported statement cannot compensate for
an unsupported one.

## Exact and semantic checks

Exact comparisons bind values to the relevant entity and attribute. Identifiers
retain case, punctuation, leading zeroes and Unicode distinctions. Quantities
use separate numeric and unit handling. Discrete values also constrain semantic
comparisons so a high similarity score cannot erase a detected literal mismatch.
Conflicts within one retrieved record are considered alongside conflicts across
records.

Semantic checks use the original evidence as the premise and the original
response statement as the hypothesis. They do not rewrite assertions into
synthetic past-tense sentences. Confirmed exact comparisons do not receive an
unrelated semantic veto. Relevant incompletely interpreted competing evidence
remains unresolved.

Reports include evidence IDs, the method used, and optional NLI scores.
Unevaluated statements do not receive fabricated zero probabilities. Coverage
is the fraction of factual statements supported; it is distinct from an NLI
probability. The host's support threshold applies to semantic verification.

## Repair, failure, and persistence

A failed check requests at most one repair with specific claim feedback. The
repaired answer is checked against the original evidence and any additional
eligible evidence obtained during repair. The final answer receives an explicit
uncertainty caveat when claims remain unresolved, verification fails, repair
fails or is interrupted, or work stops before checking finishes.

The visible answer, transcript, and checkpoint carry the same final caveat.
Device-local journals mark output awaiting grounding. Failed or interrupted
repairs reconcile durable progress immediately, preserving completed actions in
history and the next-turn checkpoint. Recovery caveats unfinished output; it does not recover a
rejected draft as an uncaveated final answer or automatically replay actions.
Previously committed canonical events remain immutable.

Response, evidence, and inference workloads are bounded. Exceeding a checking
budget is reported as incomplete verification. The native NLI model rejects
pairs beyond its 512-token input limit instead of silently dropping text and
scoring a partial pair. Invalid thresholds, non-finite scores and malformed
model results cannot produce a successful check.

## Scope and validation

Grounding establishes support relative to available sources. It does not make a
memory authoritative or current, settle ambiguous references automatically, or
guarantee that a probabilistic language model understands every statement.
The semantic model is trained for English; exact string comparisons preserve
Unicode data. Tool permissions and action authorization remain separate from
answer checking.

The test suite covers deterministic report behavior, negative and mixed claims,
literal and numeric distinctions, evidence scoping, real harness repair,
interruption, journal recovery, and compatibility with older serialized records.
Optional local-model tests compare native inference against pinned reference
logits and exercise token-limit rejection:

```bash
SPINE_NLI_DIR=/absolute/path/to/nli-MiniLM2-L6-H768 \
  cargo test --locked -p spine-models --test nli_reference
```

`grounding_replay` evaluates fixed synthetic response/evidence fixtures through
the production verifier. `grounding_live_probe` runs a small synthetic
generation-and-repair check through a configured provider, without action tools
or a persistent heart. It reads credentials from `SPINE_LLM_API_KEY` and does not
write them into its results.

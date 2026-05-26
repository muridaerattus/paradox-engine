# Classpect Experiment Report — Part 4: Deep Diagnostics

## Attempt 10: Ranked Top-K And Abstention Diagnostics

### Method

This experiment tested whether local Qwen was near the right labels even when exact top-1 predictions failed. A new ranked evaluator asks for up to three class candidates, up to three aspect candidates, and an abstention flag from each anonymous description.

The script is:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.evaluation.evaluate_ranked_classpect --descriptions-path <descriptions.json> --run-name <run_name> --model Qwen3.6-35B-A3B-MTP-GGUF --max-tokens 384 --request-timeout 300 --resume
```

Runs completed for:

```text
ranked_v1
ranked_v2
ranked_v3
```

Saved reports:

```text
experiments/classpect/generated/ranked_classpect_report_ranked_v1.json
experiments/classpect/generated/ranked_classpect_report_ranked_v2.json
experiments/classpect/generated/ranked_classpect_report_ranked_v3.json
```

### Results

| Dataset | Rows | Class@1 | Class@2 | Class@3 | Aspect@1 | Aspect@2 | Aspect@3 | Title@1 | Title@2 | Title@3 | Abstain | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 baseline | 147 | 0.163 | 0.286 | 0.408 | 0.163 | 0.238 | 0.293 | 0.014 | 0.054 | 0.095 | 0.000 | 0.000 |
| v2 contrast | 145 | 0.193 | 0.283 | 0.338 | 0.076 | 0.145 | 0.248 | 0.014 | 0.034 | 0.076 | 0.000 | 0.000 |
| v3 aspect-balanced | 136 | 0.162 | 0.250 | 0.324 | 0.059 | 0.191 | 0.316 | 0.007 | 0.044 | 0.125 | 0.000 | 0.000 |

### Outcome

Top-k ranking helped class somewhat, especially on the v1 baseline where class@3 reached `0.408`. It did not reveal a strong hidden signal for full titles: title@3 stayed at or below `0.125` across all datasets. Aspect remained the limiting factor, and the model never used the abstention flag.

This means the failures are not merely top-1 near misses. Local Qwen often fails to rank the true aspect in the top three. Do not build a reranker yet, and do not run MIPRO on v1/v2/v3 based on these top-k results.

The next useful work is either to inspect invalid-generation/leakage patterns to improve local dataset generation reliability, or to prepare the final controlled Claude Haiku vs local Qwen comparison once Anthropic credits are available. Because the model comparison is intended to be last and Anthropic is currently blocked by billing, the remaining local-only diagnostic is invalid-generation analysis.

## Attempt 15: Pairwise Confusion Audit

### Method

The pairwise audit tested whether the model can recover the true label when the candidate set is restricted to one repeated confusion pair. This uses the existing v1 baseline generated descriptions only; no new descriptions, non-canonical examples, or leaked identifiers were added.

The evaluator asks for exactly one of `A`, `B`, `Tie`, or `Abstain` for each applicable generated description. It reports true-label wins, sink-label wins, ties, abstentions, and parse/request errors separately for class and aspect.

The run used:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.evaluation.evaluate_pairwise_classpect --run-name pairwise_v1_confusions --resume --max-tokens 32 --request-timeout 300
```

Report written to:

```text
experiments/classpect/generated/pairwise_classpect_report_pairwise_v1_confusions.json
```

Dataset fingerprint:

```text
63e997f24372b57ef72df61470bc56626d248979d22324f79203638412a662a0
```

### Results

Overall pairwise recovery on the tested confusion rows was `149/256 = 0.582`. There were no ties, abstentions, or prediction errors.

| Target | Rows | True-Label Win | Sink-Label Win | Tie | Abstain | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Class | 127 | 0.606 | 0.394 | 0.000 | 0.000 | 0.000 |
| Aspect | 129 | 0.558 | 0.442 | 0.000 | 0.000 | 0.000 |

Pair-level results:

| Target | Pair | Rows | True-Label Win | Sink-Label Win | Error |
| --- | --- | ---: | ---: | ---: | ---: |
| Aspect | `Breath` vs `Hope` | 30 | 0.800 | 0.200 | 0.000 |
| Aspect | `Light` vs `Mind` | 23 | 0.478 | 0.522 | 0.000 |
| Aspect | `Space` vs `Life` | 27 | 0.444 | 0.556 | 0.000 |
| Aspect | `Space` vs `Light` | 27 | 0.593 | 0.407 | 0.000 |
| Aspect | `Time` vs `Doom` | 22 | 0.409 | 0.591 | 0.000 |
| Class | `Heir` vs `Knight` | 29 | 0.655 | 0.345 | 0.000 |
| Class | `Page` vs `Knight` | 29 | 0.552 | 0.448 | 0.000 |
| Class | `Seer` vs `Mage` | 24 | 0.583 | 0.417 | 0.000 |
| Class | `Thief` vs `Prince` | 22 | 0.682 | 0.318 | 0.000 |
| Class | `Thief` vs `Knight` | 23 | 0.565 | 0.435 | 0.000 |

### Outcome

The audit does not justify building a full 12-way tournament. Class pairwise recovery barely clears the weak `0.60` threshold, while aspect pairwise recovery is below it and several aspect pairs are worse than chance. That means the model often cannot recover the correct aspect even when the choice is reduced to two labels.

The current anonymous-description classifier path should be treated as underdetermined for aspect and full-title recovery. Further work should change the representation or supervision target rather than expanding prompt-only generation, running another constrained/direct MIPRO pass, or building a full tournament on top of weak pairwise signal.


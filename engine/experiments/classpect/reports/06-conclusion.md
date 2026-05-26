# Classpect Experiment Report — Part 6: Conclusion

## Current Recommendation

Use the promoted separate class/aspect DSPy programs as the current prompt source:

```text
experiments/classpect/generated/dspy_class_program.json
experiments/classpect/generated/dspy_aspect_program.json
```

Do not use the joint title artifacts:

```text
experiments/classpect/generated/dspy_classpect_program.json
experiments/classpect/generated/dspy_classpect_program_fold_0.json
```

Do not promote the final all-data bootstrap artifacts, the no-think bootstrap artifacts, the constrained direct/no-think bootstrap artifacts, the clean nested constrained direct/no-think MIPRO artifacts, or the contrast-v2/v3 direct/no-think artifacts. The final artifacts fit poorly even on train data, and the no-think/v2/v3/grouped nested-MIPRO runs underperformed the current promoted separate-program baseline.

The current clean no-name results are worse than earlier leaked/stale numbers, but they are more honest. The main value now is that evaluation is grouped by character, variants do not leak between train and dev, saved artifacts are fingerprinted, and output failures are tracked instead of crashing or silently disappearing.

## Suggestions From Current Results

We are intentionally not adding an external/reserved evaluation set because there is no extra canonical data to reserve. We are also not adding non-canonical synthetic examples; the experiment should remain grounded in the 32 canon-labeled examples only.

### General Principles

| Principle | Reason |
| --- | --- |
| Keep promoted fold-selected prompts as the current best artifacts | They still outperform final bootstrap, no-think bootstrap, prompt-regeneration, Haiku-generation, and clean nested constrained/direct MIPRO comparisons. |
| Change one major variable at a time | If prompt style, optimizer, model, and descriptions all change together, results become hard to interpret. |
| Treat model choice as a major variable | The local Qwen runs may not be directly comparable to earlier Claude Haiku runs; model effects can look like prompt or data regressions. |
| Use grouped CV as the performance estimate | With only canon labels available, grouped CV is the cleanest way to avoid variant leakage and compare settings. |
| Treat final all-data runs as artifact generation, not evaluation | Final train-fit metrics are not held-out estimates and can be misleading. |
| Prefer stable output before expensive optimization | Qwen parse failures and invalid labels distort accuracy; reliability should be measured beside accuracy. |
| Run MIPRO for serious prompt candidates, but nest it | MIPRO is the real optimizer; clean evaluation requires an inner grouped validation split and untouched outer dev fold. |
| Keep prediction error rates in every report | Invalid labels such as `Justice` are model compliance failures and should be visible beside accuracy. |
| Do not add non-canonical synthetic examples | The experiment should remain grounded in the 32 canon-labeled examples only. |
| Do not reserve external data that does not exist | There is no extra canonical evaluation set to hold back. |

### Actionable Research Ideas

| Priority | Idea | Why It Is Next-Worthy |
| ---: | --- | --- |
| 1 | Stop generated anonymous dossier variants | Both personality-only and abstract-role theory induction failed, and abstract-role generation made aspect accuracy collapse to `0.038`. |
| 2 | Move to an explicitly auditable representation | Future work should use hand-curated canon evidence, human-authored features, or binary questions instead of more generated summaries. |
| 3 | Keep promoted separate class/aspect artifacts | They remain the best available prompt artifacts despite the clean evaluation limits. |
| 4 | Stop prompt-only dataset iteration | V2, v3, retry-smoke, Haiku generation, and clean nested MIPRO did not produce a stronger candidate. |
| 5 | Change representation or supervision before more optimization | Future experiments need a new information structure, not more prompt wording over the same anonymous descriptions. |

## Next Experiment Plan

The ranked top-k experiment did not reveal strong hidden title signal. Prompt-only description regeneration with local Qwen failed twice on aspect discrimination, retry hardening improved hygiene without accuracy, the final Claude Haiku generation comparison improved validation compliance without improving downstream aspect/title accuracy, clean nested MIPRO did not rescue the stable constrained direct/no-think prompt, and pairwise aspect recovery stayed below the stop threshold.

### Immediate Experiment: Stop Generated-Dossier Branch

1. Do not build the full class/aspect tournament from this pairwise evaluator; the pairwise signal is too weak, especially for aspect.
2. Do not continue optimizing personality-only theory induction; it underperformed the promoted artifacts.
3. Do not continue optimizing abstract-role theory induction; it underperformed personality-only theory induction and nearly collapsed aspect recovery.
4. Do not generate another anonymous-dossier variant unless the representation is no longer freeform LLM summary text.
5. If continuing, design an auditable evidence instrument: fixed binary/ordinal questions, a human-authored feature matrix, or hand-curated canon snippets with explicit no-leak review.
6. Keep the anonymous/no-leak policy and grouped-by-character CV as non-negotiable evaluation constraints.
7. Treat any future full-data artifact as deployment packaging only, not as evidence of generalization.

### Decision Gates

1. Do not promote Haiku, v2, v3, retry-smoke, constrained direct/no-think bootstrap, clean nested constrained/direct MIPRO, joint title, or final all-data bootstrap artifacts.
2. Do not use MIPRO results as clean CV estimates unless MIPRO uses an inner grouped validation split and the outer dev fold remains untouched.
3. Pairwise v1 confusion recovery was `0.582` overall, with aspect at `0.558`; do not expand this into a full tournament.
4. Personality-only fold-local theory induction reached class `0.197`, aspect `0.170`, and title `0.020`; do not promote or optimize it further.
5. Abstract-role fold-local theory induction reached class `0.125`, aspect `0.038`, and title `0.000`; do not promote or optimize it further.
6. Only run another full generation/evaluation cycle after changing away from generated freeform summaries toward an auditable representation or supervision signal.

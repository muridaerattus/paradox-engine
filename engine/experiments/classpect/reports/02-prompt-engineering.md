# Classpect Experiment Report — Part 2: Prompt Engineering

## Attempt 4: Profile-Guided Rubrics

### Method

This experiment first generated 2-3 mini-descriptions per class and aspect by asking what anonymous descriptions with the same label have in common. The classifier then received those mini-descriptions as a rubric.

For leakage-safe CV, profiles were generated separately per fold from train characters only:

```bash
for fold in 0 1 2 3; do
  PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.generate_label_profiles --fold "$fold" --num-threads 4
done
```

### Local Qwen Setup

The local model is reached via the OpenAI-compatible API base read from `LOCAL_LLM_API_BASE`. Qwen returned long hidden `reasoning_content` unless thinking was disabled, so `configure_dspy` now passes:

```python
chat_template_kwargs={"enable_thinking": False}
```

The profile-guided modules use direct `dspy.Predict` rather than `dspy.ChainOfThought` to reduce output length and parse failures. The main separate class/aspect DSPy modules remain chain-of-thought to preserve compatibility with the promoted existing artifacts.

### Results

Unoptimized profile-rubric baseline:

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.167 |
| Mean dev aspect accuracy | 0.131 |
| Mean dev title accuracy | 0.014 |
| Mean dev vote class accuracy | 0.156 |
| Mean dev vote aspect accuracy | 0.156 |
| Mean dev vote title accuracy | 0.000 |

Local Qwen profile-guided bootstrap run used:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.train_profile_guided_dspy_classpect --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --skip-train-eval --num-threads 8 --max-tokens 1024 --save
```

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.162 |
| Mean dev aspect accuracy | 0.211 |
| Mean dev title accuracy | 0.000 |
| Mean dev vote class accuracy | 0.156 |
| Mean dev vote aspect accuracy | 0.188 |
| Mean dev vote title accuracy | 0.000 |

Fold-level profile-guided bootstrap metrics:

| Fold | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.171 | 0.257 | 0.000 | 0.125 | 0.250 | 0.000 |
| 1 | 0.216 | 0.081 | 0.000 | 0.250 | 0.000 | 0.000 |
| 2 | 0.231 | 0.256 | 0.000 | 0.250 | 0.250 | 0.000 |
| 3 | 0.028 | 0.250 | 0.000 | 0.000 | 0.250 | 0.000 |

### Deterministic Diagnostics

Two no-LLM similarity baselines were added to separate rubric quality from LLM behavior.

Profile-text similarity compares each held-out description to the generated label mini-descriptions:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.evaluate_profile_similarity
```

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.054 |
| Mean dev aspect accuracy | 0.140 |
| Mean dev title accuracy | 0.007 |
| Mean dev vote class accuracy | 0.031 |
| Mean dev vote aspect accuracy | 0.156 |
| Mean dev vote title accuracy | 0.000 |

Description nearest-neighbor similarity compares each held-out description to train-fold descriptions directly:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.evaluate_description_similarity
```

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.082 |
| Mean dev aspect accuracy | 0.042 |
| Mean dev title accuracy | 0.000 |
| Mean dev vote class accuracy | 0.094 |
| Mean dev vote aspect accuracy | 0.031 |
| Mean dev vote title accuracy | 0.000 |

### Outcome

The profile-guided rubric did not improve classification. The deterministic profile-similarity baseline is near chance, which suggests the mini-description compression loses discriminative signal. The raw description nearest-neighbor baseline is also near chance, so simple lexical similarity is not enough to separate the anonymous descriptions by class/aspect.

This path is useful diagnostically, but it should not replace the promoted separate class/aspect DSPy programs.

## Attempt 6: Qwen No-Think Output Stability

### Method

This comparison isolated the Qwen output-shape problem before spending time on more expensive optimization.

Implemented switches:

| Switch | Purpose |
| --- | --- |
| `--qwen-no-think` | Adds Qwen's explicit `/no_think` flag to the DSPy signatures. |
| `--predictor direct` | Uses direct `dspy.Predict` class/aspect modules instead of `dspy.ChainOfThought`. |
| `--run-name` | Writes separate generated report/program filenames so comparison runs do not overwrite promoted artifacts. |

The cheap grouped-CV comparisons used bootstrap optimization with train evaluation skipped:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 512 --skip-train-eval --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 512 --skip-train-eval --save
```

### Results

Train metrics are intentionally zero in these reports and should be ignored.

| Run | Predictor | `/no_think` | Mean Dev Class | Mean Dev Aspect | Mean Dev Title | Mean Vote Class | Mean Vote Aspect | Mean Vote Title | Mean Dev Error Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_no_think` | direct `Predict` | yes | 0.139 | 0.239 | 0.026 | 0.156 | 0.219 | 0.031 | 0.000 |
| `no_think` | `ChainOfThought` | yes | 0.114 | 0.177 | 0.020 | 0.125 | 0.188 | 0.031 | 0.006 |

Outcome:

| Finding | Interpretation |
| --- | --- |
| Direct/no-think eliminated prediction errors | This is the most stable Qwen output shape. |
| Direct/no-think underperformed the promoted separate-program CV result | Stability alone is not enough; the direct bootstrap prompt is not a better classifier. |
| Chain-of-thought/no-think still produced truncation warnings and one dev prediction error | The explicit `/no_think` flag does not fully solve long CoT behavior for this setup. |
| Neither cheap no-think run should be promoted | Both have substantially lower vote metrics than the current promoted separate prompts. |

Direct/no-think should remain available as the stable-output path, but direct/no-think MIPRO is not justified yet unless the goal is specifically reliability rather than accuracy.

## Attempt 7: Constrained Direct/No-Think Labels

### Method

This experiment kept the stable direct/no-think path fixed and tightened only the output contract. The new constrained signatures require exactly one valid label and no explanation, punctuation, qualifiers, aliases, or invented labels.

The grouped-CV run used bootstrap optimization with train evaluation skipped:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --predictor direct --qwen-no-think --constrained-labels --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name direct_no_think_constrained
```

The first run timed out during fold 3 evaluation after folds 0-2 completed, then finished with `--resume`.

### Results

Train metrics are intentionally zero in this report and should be ignored.

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.133 |
| Mean dev aspect accuracy | 0.250 |
| Mean dev title accuracy | 0.033 |
| Mean dev vote class accuracy | 0.156 |
| Mean dev vote aspect accuracy | 0.281 |
| Mean dev vote title accuracy | 0.031 |
| Mean dev prediction error rate | 0.000 |

Fold-level constrained direct/no-think metrics:

| Fold | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title | Dev Error Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.029 | 0.371 | 0.029 | 0.000 | 0.500 | 0.000 | 0.000 |
| 1 | 0.135 | 0.108 | 0.000 | 0.125 | 0.125 | 0.000 | 0.000 |
| 2 | 0.256 | 0.436 | 0.103 | 0.375 | 0.375 | 0.125 | 0.000 |
| 3 | 0.111 | 0.083 | 0.000 | 0.125 | 0.125 | 0.000 | 0.000 |

Aggregate prediction counts showed broad class spread but aspect concentration on `Hope` and `Mind`:

| Target | Prediction Counts |
| --- | --- |
| Dev class | `Bard=10`, `Heir=1`, `Knight=31`, `Mage=16`, `Maid=16`, `Page=9`, `Prince=25`, `Rogue=9`, `Seer=7`, `Sylph=10`, `Thief=2`, `Witch=11` |
| Dev aspect | `Blood=3`, `Breath=6`, `Doom=9`, `Heart=10`, `Hope=36`, `Life=15`, `Light=9`, `Mind=30`, `Rage=15`, `Space=1`, `Time=3`, `Void=10` |
| Vote class | `Bard=3`, `Heir=1`, `Knight=5`, `Mage=5`, `Maid=3`, `Page=2`, `Prince=5`, `Rogue=2`, `Seer=1`, `Sylph=2`, `Witch=3` |
| Vote aspect | `Blood=1`, `Breath=2`, `Doom=2`, `Heart=2`, `Hope=7`, `Life=3`, `Light=1`, `Mind=7`, `Rage=3`, `Time=1`, `Void=3` |

### Outcome

The constrained prompt preserved the main reliability benefit of direct/no-think: prediction error rate stayed at zero. It did not improve accuracy. It roughly matched the existing direct/no-think vote class and vote title metrics, improved vote aspect from `0.219` to `0.281`, but still remained far below the promoted separate class/aspect CV baseline.

Do not promote constrained direct/no-think artifacts, and do not spend MIPRO budget on this constrained prompt. The next effort should move to higher-quality anonymous descriptions while keeping the same grouped-CV comparison structure.

One important caveat: these newer no-think experiments ran on the local Qwen model, while some earlier work used Claude Haiku before Anthropic usage limits interrupted evaluation. Worse performance may therefore reflect a model change, not only the stricter no-name data, direct/no-think prompting, or constrained output format. However, the model-swap comparison should be done last, after cheaper local-only prompt, diagnostics, and description-quality work has been exhausted.

## Attempt 14: Clean Nested MIPRO For Stable Direct/No-Think Prompt

### Method

The previous recommendation to delay MIPRO was too conservative. Bootstrap/direct/no-think results are useful smoke tests for output stability, but they are not a fair final judgment of a DSPy prompt shape because MIPRO is the actual prompt optimizer.

One evaluation issue had to be fixed first: the existing MIPRO path passed the outer fold dev set as MIPRO's `valset`, then evaluated on that same dev set. That makes the result useful for prompt selection but not a clean held-out estimate. The trainer now supports an inner grouped validation split for MIPRO:

```text
--mipro-inner-folds 4
```

For each outer grouped fold, the outer train characters are split again by observed character id. MIPRO optimizes on the inner train/validation split, and the outer dev characters remain untouched until final evaluation.

The run used the v1 baseline descriptions and the most stable local-Qwen output shape:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --predictor direct --qwen-no-think --constrained-labels --optimizer mipro --mipro-inner-folds 4 --num-trials 0 --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 512 --skip-train-eval --save --resume --run-name v1_constrained_direct_no_think_mipro_inner
```

`--num-trials 0` uses DSPy's light-mode MIPRO setting, which ran 10 trials per optimized module. Train metrics are intentionally zero and should be ignored.

Reports were written to:

```text
experiments/classpect/generated/dspy_classpect_report_v1_constrained_direct_no_think_mipro_inner.json
experiments/classpect/generated/dspy_classpect_diagnostics_v1_constrained_direct_no_think_mipro_inner.json
```

### Results

| Run | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constrained direct/no-think bootstrap | 0.133 | 0.250 | 0.033 | 0.156 | 0.281 | 0.031 | 0.000 |
| clean nested MIPRO, constrained direct/no-think | 0.191 | 0.187 | 0.038 | 0.219 | 0.156 | 0.031 | 0.000 |
| promoted separate class/aspect MIPRO artifacts | 0.324 | 0.394 | 0.127 | 0.281 | 0.375 | 0.125 | historical clean CV |

Fold-level clean nested MIPRO metrics:

| Fold | Train Size | Dev Size | Inner Train | Inner Val | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 112 | 35 | 86 | 26 | 0.000 | 0.143 | 0.000 | 0.000 | 0.125 | 0.000 |
| 1 | 110 | 37 | 81 | 29 | 0.270 | 0.108 | 0.000 | 0.250 | 0.125 | 0.000 |
| 2 | 108 | 39 | 79 | 29 | 0.385 | 0.359 | 0.154 | 0.500 | 0.250 | 0.125 |
| 3 | 111 | 36 | 87 | 24 | 0.111 | 0.139 | 0.000 | 0.125 | 0.125 | 0.000 |

Clean MIPRO improved the constrained direct/no-think class metrics relative to bootstrap, but aspect and vote-aspect regressed. It still did not approach the promoted separate class/aspect MIPRO artifacts.

### Diagnostics

Top recurring clean nested MIPRO mistakes included:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Heir -> Knight`, `Page -> Knight`, `Knight -> Bard`, `Sylph -> Maid`, `Mage -> Bard`, `Maid -> Seer`, `Thief -> Knight`, `Bard -> Witch` |
| Aspect | `Light -> Mind`, `Breath -> Hope`, `Life -> Hope`, `Rage -> Void`, `Space -> Light`, `Time -> Doom`, `Heart -> Hope`, `Life -> Rage` |

Prediction bias remained concentrated around familiar sinks:

| Target | Largest Over-Predictions | Largest Under-Predictions |
| --- | --- | --- |
| Class | `Knight=+27`, `Bard=+17`, `Mage=+4`, `Seer=+4` | `Rogue=-13`, `Heir=-12`, `Sylph=-8`, `Thief=-8` |
| Aspect | `Mind=+25`, `Hope=+20`, `Rage=+7`, `Doom=+6` | `Space=-14`, `Breath=-12`, `Time=-11`, `Light=-7`, `Void=-7` |

### Outcome

This corrects the earlier MIPRO recommendation: MIPRO should not be delayed merely because bootstrap is weak. It should be run when testing a serious prompt shape, but it must use an inner validation split so outer grouped-CV remains clean.

For this specific stable constrained direct/no-think prompt, clean MIPRO did not produce a promotable artifact. It improved class over constrained bootstrap but failed the aspect/title gate and stayed below the promoted separate class/aspect artifacts.

The next direction should not be another prompt-only generation run or another constrained direct/no-think MIPRO run. The next useful test is a pairwise/tournament audit over the repeated confusion pairs to determine whether the model can distinguish the correct label from its common sink labels when the candidate set is small.


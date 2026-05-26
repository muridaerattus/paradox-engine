# Classpect Experiment Report — Part 3: Description Dataset Iteration

## Experiment Structure: Diagnostics Artifacts

The experiment now has a reusable diagnostics pass for any saved DSPy report:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.diagnose_dspy_report --run-name direct_no_think --limit 8 --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.diagnose_dspy_report --run-name direct_no_think_constrained --limit 8 --save
```

This writes diagnostics JSON next to the run reports:

```text
experiments/classpect/generated/dspy_classpect_diagnostics_direct_no_think.json
experiments/classpect/generated/dspy_classpect_diagnostics_direct_no_think_constrained.json
```

The diagnostics include:

| Diagnostic | Purpose |
| --- | --- |
| Top class confusions | Shows repeated `expected -> predicted` class mistakes and the characters involved. |
| Top aspect confusions | Shows repeated `expected -> predicted` aspect mistakes and the characters involved. |
| Vote-level confusions | Filters variant noise and shows character-majority mistakes. |
| Prediction bias | Shows over-predicted and under-predicted labels by count delta. |

The two direct/no-think runs have similar top confusion structure, which suggests the constrained output contract changed compliance more than discrimination. Repeated class confusions include `Heir -> Knight`, `Seer -> Mage`, `Thief -> Prince`, `Maid -> Mage`, `Page -> Knight`, and `Rogue -> Knight`. Repeated aspect confusions include `Light -> Mind`, `Breath -> Hope`, `Space -> Life`, `Space -> Light`, `Life -> Hope`, and `Doom -> Mind`.

The constrained run shifted aspect prediction bias toward `Hope` and `Mind`:

| Run | Largest Aspect Over-Predictions | Largest Aspect Under-Predictions |
| --- | --- | --- |
| `direct_no_think` | `Mind=+19`, `Light=+8`, `Rage=+8`, `Doom=+4` | `Space=-14`, `Time=-10`, `Void=-9`, `Breath=-8` |
| `direct_no_think_constrained` | `Hope=+21`, `Mind=+20`, `Rage=+7`, `Life=+2` | `Space=-13`, `Breath=-9`, `Time=-9`, `Blood=-5`, `Void=-5` |

Use these diagnostics before writing a new prompt or regenerating descriptions. The next useful description-generation change should specifically sharpen the repeated confusion pairs instead of broadly asking for better summaries.

## Experiment Structure: Description Dataset Versions

The description generator now supports a contrastive v2 prompt and versioned output paths. The contrast targets are tracked in:

```text
experiments/classpect/reports/CONTRAST_TARGETS.md
```

Generate a v2 description dataset without overwriting the baseline `descriptions.json`:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --prompt-style contrast-v2 --variants 5 --run-name contrast_v2
```

Anthropic generation was blocked by billing during this run:

```text
Your credit balance is too low to access the Anthropic API.
```

The generator therefore also supports the local OpenAI-compatible Qwen backend:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --model Qwen3.6-35B-A3B-MTP-GGUF --prompt-style contrast-v2 --variants 5 --run-name contrast_v2 --concurrency 2 --max-retries 2 --max-tokens 1536 --request-timeout 300 --resume
```

If a run produces invalid records, resume while discarding invalid rows so only failed variants are regenerated:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --model Qwen3.6-35B-A3B-MTP-GGUF --prompt-style contrast-v2 --variants 5 --run-name contrast_v2 --concurrency 2 --max-retries 3 --max-tokens 1536 --request-timeout 300 --resume --retry-invalid
```

This writes:

```text
experiments/classpect/generated/descriptions_contrast_v2.json
```

Summarize and fingerprint the generated dataset before training:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.summarize_descriptions --descriptions-path experiments/classpect/generated/descriptions_contrast_v2.json
```

The DSPy trainer can now read an explicit description dataset path:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_contrast_v2.json --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name contrast_v2_direct_no_think
```

Use a separate run name for constrained comparison if needed:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_contrast_v2.json --predictor direct --qwen-no-think --constrained-labels --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name contrast_v2_direct_no_think_constrained
```

After each v2 run, write diagnostics with:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.diagnose_dspy_report --run-name contrast_v2_direct_no_think --limit 8 --save
```

## Attempt 8: Diagnostic-Guided Description v2
## Attempt 8: Diagnostic-Guided Description v2

### Method

This experiment generated a contrastive v2 anonymous description dataset using the local Qwen OpenAI-compatible backend because Anthropic credits were unavailable. The v2 prompt emphasized contrastive personality evidence guided by `CONTRAST_TARGETS.md` while keeping the same no-name/no-metadata validation policy.

Dataset summary:

| Item | Count |
| --- | ---: |
| Generated records | 160 |
| Valid records after filtering | 145 |
| Invalid records filtered out | 15 |
| Characters represented | 32 |
| Minimum valid variants per character | 1 |
| Maximum valid variants per character | 5 |

Dataset fingerprint:

```text
1c9ccc82513bb63f6e6434be76a864a367e5a127812847d1053535ee5a4c6126
```

The grouped-CV run used cheap local-Qwen direct/no-think bootstrap settings:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_contrast_v2.json --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name contrast_v2_direct_no_think
```

### Results

Train metrics are intentionally zero and should be ignored.

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.176 |
| Mean dev aspect accuracy | 0.058 |
| Mean dev title accuracy | 0.023 |
| Mean dev vote class accuracy | 0.156 |
| Mean dev vote aspect accuracy | 0.094 |
| Mean dev vote title accuracy | 0.031 |
| Mean dev prediction error rate | 0.000 |

Fold-level v2 direct/no-think metrics:

| Fold | Train Size | Dev Size | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 106 | 39 | 0.000 | 0.026 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 109 | 36 | 0.056 | 0.028 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 113 | 32 | 0.438 | 0.125 | 0.094 | 0.375 | 0.250 | 0.125 |
| 3 | 107 | 38 | 0.211 | 0.053 | 0.000 | 0.250 | 0.125 | 0.000 |

Diagnostics were written to:

```text
experiments/classpect/generated/dspy_classpect_diagnostics_contrast_v2_direct_no_think.json
```

Top recurring v2 mistakes included:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Page -> Knight`, `Sylph -> Seer`, `Prince -> Knight`, `Witch -> Seer`, `Heir -> Knight`, `Maid -> Mage` |
| Aspect | `Doom -> Mind`, `Heart -> Life`, `Light -> Mind`, `Blood -> Rage`, `Breath -> Rage`, `Space -> Light`, `Void -> Blood` |

Prediction bias worsened for aspect. V2 over-predicted `Mind` by `+38` raw rows and `+10` vote rows, while under-predicting `Breath`, `Time`, `Space`, and `Heart`.

### Outcome

The v2 dataset did not pass the decision gate. It improved raw dev class accuracy relative to v1 direct/no-think (`0.176` vs `0.139`) but collapsed raw dev aspect accuracy (`0.058` vs `0.239`) and vote aspect accuracy (`0.094` vs `0.219`). Prediction error rate stayed at zero, so this was a discrimination failure rather than an output-shape failure.

Do not run constrained comparison or MIPRO on this v2 dataset. The next local-only step should revise the v2 description prompt to reduce aspect over-concentration, especially the `Mind` sink, before generating another dataset version.

## Attempt 9: Aspect-Balanced Description v3

### Method

This experiment made one local-only change after v2: the contrastive description prompt was revised to avoid over-focusing on analysis, judgment, decision logic, planning, and consequences. It explicitly asked for recoverable evidence for under-predicted aspect patterns such as independence/detachment, timing/pressure, space/environment/creation, and emotional interiority.

The generation command was:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --model Qwen3.6-35B-A3B-MTP-GGUF --prompt-style contrast-v3 --variants 5 --run-name contrast_v3 --concurrency 2 --max-retries 2 --max-tokens 1536 --request-timeout 300 --resume
```

Dataset summary:

| Item | Count |
| --- | ---: |
| Generated records | 160 |
| Valid records after filtering | 136 |
| Invalid records filtered out | 24 |
| Characters represented | 32 |
| Minimum valid variants per character | 1 |
| Maximum valid variants per character | 5 |

Dataset fingerprint:

```text
4674b1044af2ccdb9d41f4c4f51572d7b103c86b406b06405e789d17286dc44a
```

The grouped-CV run used the same cheap local-Qwen direct/no-think bootstrap settings:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_contrast_v3.json --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name contrast_v3_direct_no_think
```

### Results

Train metrics are intentionally zero and should be ignored.

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.145 |
| Mean dev aspect accuracy | 0.029 |
| Mean dev title accuracy | 0.007 |
| Mean dev vote class accuracy | 0.156 |
| Mean dev vote aspect accuracy | 0.000 |
| Mean dev vote title accuracy | 0.000 |
| Mean dev prediction error rate | 0.000 |

Fold-level v3 direct/no-think metrics:

| Fold | Train Size | Dev Size | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 104 | 32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 104 | 32 | 0.156 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 |
| 2 | 102 | 34 | 0.265 | 0.088 | 0.029 | 0.375 | 0.000 | 0.000 |
| 3 | 98 | 38 | 0.158 | 0.026 | 0.000 | 0.125 | 0.000 | 0.000 |

Diagnostics were written to:

```text
experiments/classpect/generated/dspy_classpect_diagnostics_contrast_v3_direct_no_think.json
```

Top recurring v3 mistakes included:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Page -> Knight`, `Witch -> Bard`, `Witch -> Knight`, `Bard -> Maid`, `Bard -> Witch`, `Heir -> Maid`, `Rogue -> Knight`, `Thief -> Prince` |
| Aspect | `Space -> Doom`, `Hope -> Life`, `Breath -> Life`, `Light -> Mind`, `Blood -> Rage`, `Breath -> Mind`, `Mind -> Breath`, `Rage -> Life` |

V3 reduced the single `Mind` sink compared with v2, but it created broader aspect sink behavior: `Mind=+18`, `Doom=+11`, `Life=+11`, and `Rage=+7` raw-row over-predictions. Vote-level aspect accuracy fell to zero.

### Outcome

The v3 dataset also failed the decision gate. The attempted aspect balancing reduced raw `Mind` over-prediction compared with v2, but it did not improve accuracy and instead moved errors into `Doom`, `Life`, and `Rage`. V3 underperformed both v1 direct/no-think and v2 direct/no-think.

Do not run constrained comparison, repeated CV, or MIPRO on v3. At this point, prompt-only regeneration with local Qwen has produced worse aspect evidence twice. The next local-only step should switch from generating new descriptions to evaluating whether the classifier can recover the correct label among multiple candidates, using top-k or abstention metrics on existing v1/v2/v3 reports.

## Attempt 11: Invalid-Generation Analysis And Retry Hardening

### Method

A new invalid-description diagnostic script analyzes generated description records with validation errors by forbidden term and character:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.analyze_invalid_descriptions --descriptions-path <descriptions.json>
```

The script was run on the baseline, v2, and v3 local-Qwen description datasets.

### Results

| Dataset | Generated | Invalid | Invalid Rate | Top Forbidden Terms | Top Invalid Characters |
| --- | ---: | ---: | ---: | --- | --- |
| v1 baseline | 160 | 13 | 0.081 | `troll=9`, `weapon=3`, `human=1`, `powers=1` | `Damara Megido=3`, `Jane Crocker=2` |
| v2 contrast | 160 | 15 | 0.094 | `human=10`, `troll=6`, `weapon=6` | `Cronus Ampora=4`, `Latula Pyrope=4`, `Rose Lalonde=3` |
| v3 aspect-balanced | 160 | 24 | 0.150 | `weapon=17`, `human=12`, `troll=7` | `Aradia Megido=4`, `Rose Lalonde=4`, `Dave Strider=3`, `Porrim Maryam=3` |

Invalid rows are concentrated around a small set of metadata terms, not a broad parser failure. V3 in particular regressed on `weapon` leakage.

The generator also had a retry weakness: with local Qwen at temperature `0`, invalid retries reused the same prompt and could reproduce the same forbidden output. Retry calls now include a rejection-specific regeneration instruction listing the exact rejected strings and requiring a fresh anonymous profile without those strings or related metadata.

### Outcome

This is a reliability fix, not evidence that v2/v3 should be promoted. The existing v2/v3 accuracy failures remain valid. If another local dataset attempt is made, it should first run a small retry-hardened smoke generation and check whether invalid rates fall before spending a full 160-record run.

Do not run MIPRO or promote any artifact from this change alone. The retry hardening only improves generation hygiene; it does not address weak aspect/title discrimination by itself.

## Attempt 12: Retry-Hardened Smoke Dataset

### Method

A one-variant smoke dataset was generated with the retry-hardened generator to test whether invalid leakage could be reduced before any full local regeneration:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --model Qwen3.6-35B-A3B-MTP-GGUF --prompt-style baseline --variants 1 --run-name retry_smoke --concurrency 2 --max-retries 2 --max-tokens 1536 --request-timeout 300
```

The first pass still ended with `3/32` invalid records (`0.094` invalid rate), with remaining leaks around `weapon`, `human`, and `title`. Resuming with `--retry-invalid` and a higher retry budget regenerated only the invalid rows:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --model Qwen3.6-35B-A3B-MTP-GGUF --prompt-style baseline --variants 1 --run-name retry_smoke --concurrency 2 --max-retries 4 --max-tokens 1536 --request-timeout 300 --resume --retry-invalid
```

Final smoke dataset summary:

| Item | Count |
| --- | ---: |
| Generated records | 32 |
| Valid records after filtering | 32 |
| Invalid records | 0 |
| Characters represented | 32 |
| Minimum valid variants per character | 1 |
| Maximum valid variants per character | 1 |

Dataset fingerprint:

```text
9d7269497566e616f41692ff78040e3636b8da8dfed70c624908b3a8a7e49fcc
```

The smoke dataset was then evaluated with the same cheap local-Qwen direct/no-think bootstrap settings:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_retry_smoke.json --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name retry_smoke_direct_no_think
```

Diagnostics were written to:

```text
experiments/classpect/generated/dspy_classpect_diagnostics_retry_smoke_direct_no_think.json
```

### Results

Train metrics are intentionally zero and should be ignored.

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.125 |
| Mean dev aspect accuracy | 0.062 |
| Mean dev title accuracy | 0.000 |
| Mean dev vote class accuracy | 0.125 |
| Mean dev vote aspect accuracy | 0.062 |
| Mean dev vote title accuracy | 0.000 |
| Mean dev prediction error rate | 0.000 |

Fold-level retry-smoke metrics:

| Fold | Train Size | Dev Size | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 24 | 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 24 | 8 | 0.125 | 0.125 | 0.000 | 0.125 | 0.125 | 0.000 |
| 2 | 24 | 8 | 0.250 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 |
| 3 | 24 | 8 | 0.125 | 0.125 | 0.000 | 0.125 | 0.125 | 0.000 |

Top recurring retry-smoke mistakes included:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Page -> Knight`, `Prince -> Knight`, `Bard -> Mage`, `Bard -> Witch`, `Heir -> Knight`, `Heir -> Mage` |
| Aspect | `Breath -> Rage`, `Light -> Mind`, `Time -> Void`, `Blood -> Light`, `Blood -> Rage`, `Breath -> Mind` |

Prediction bias still favored `Knight` for class and `Mind`/`Rage` for aspect. The retry fix improved hygiene but did not improve classification quality.

### Outcome

Retry hardening can produce a fully valid small local-Qwen dataset after `--retry-invalid`, but the smoke dataset underperformed the existing baseline and does not justify a full regeneration, MIPRO, or artifact promotion.

The local-only prompt/dataset iteration path is now exhausted unless a new, more substantive data-generation idea is introduced. The next controlled experiment should be the final Claude Haiku vs local Qwen comparison when Anthropic credits are available, changing only the model/backend configuration.

## Attempt 13: Final Claude Haiku vs Local Qwen Generation Comparison

### Method

Anthropic credits became available, so the final controlled model comparison was run. The first attempted Haiku alias was rejected by the Anthropic API:

```text
model: claude-3-5-haiku-latest
```

The available model list included:

```text
claude-haiku-4-5-20251001
```

The Haiku dataset used the same baseline anonymous-description prompt, same five variants per character, same validation policy, and same retry-hardened generator. The changed variable was the description-generation backend/model:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend anthropic --model claude-haiku-4-5-20251001 --prompt-style baseline --variants 5 --run-name haiku_baseline --concurrency 2 --max-retries 4 --max-tokens 1536 --request-timeout 300 --resume
```

The resulting dataset was evaluated with the same local-Qwen direct/no-think bootstrap classifier settings used for the local-Qwen baseline comparisons:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --descriptions-path experiments/classpect/generated/descriptions_haiku_baseline.json --predictor direct --qwen-no-think --optimizer bootstrap --max-bootstrapped-demos 2 --max-labeled-demos 8 --num-threads 8 --max-tokens 256 --skip-train-eval --save --run-name haiku_baseline_direct_no_think
```

Diagnostics and ranked top-k reports were written to:

```text
experiments/classpect/generated/dspy_classpect_diagnostics_haiku_baseline_direct_no_think.json
experiments/classpect/generated/ranked_classpect_report_ranked_haiku_baseline.json
```

### Dataset Results

| Dataset | Generator | Generated | Valid | Invalid | Invalid Rate | Characters | Fingerprint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen v1 baseline | local Qwen | 160 | 147 | 13 | 0.081 | 32 | existing v1 baseline |
| Haiku baseline | Claude Haiku 4.5 | 160 | 160 | 0 | 0.000 | 32 | `0b2250b3ae5bb1154ea23d110a56a3f100376393cb1c957cf91714a67168c3b0` |

Haiku was clearly better at validation compliance. It produced a complete 160-row dataset with no invalid rows after retry handling.

### Grouped-CV Results

Train metrics are intentionally zero and should be ignored.

| Dataset | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen v1 baseline direct/no-think | 0.139 | 0.239 | 0.026 | 0.156 | 0.219 | 0.031 | 0.000 |
| Haiku baseline direct/no-think | 0.156 | 0.113 | 0.013 | 0.156 | 0.125 | 0.031 | 0.000 |

Fold-level Haiku metrics:

| Fold | Train Size | Dev Size | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 40 | 0.100 | 0.325 | 0.050 | 0.125 | 0.500 | 0.125 |
| 1 | 120 | 40 | 0.050 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 120 | 40 | 0.350 | 0.075 | 0.000 | 0.375 | 0.000 | 0.000 |
| 3 | 120 | 40 | 0.125 | 0.025 | 0.000 | 0.125 | 0.000 | 0.000 |

Haiku improved raw class accuracy slightly versus the local-Qwen baseline, but vote class was unchanged and aspect accuracy fell sharply. Full-title accuracy remained effectively unusable.

### Ranked Top-K Results

| Dataset | Rows | Class@1 | Class@2 | Class@3 | Aspect@1 | Aspect@2 | Aspect@3 | Title@1 | Title@2 | Title@3 | Abstain | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen v1 baseline | 147 | 0.163 | 0.286 | 0.408 | 0.163 | 0.238 | 0.293 | 0.014 | 0.054 | 0.095 | 0.000 | 0.000 |
| Haiku baseline | 160 | 0.150 | 0.325 | 0.369 | 0.131 | 0.231 | 0.312 | 0.006 | 0.075 | 0.087 | 0.000 | 0.000 |

The ranked result does not change the decision. Haiku has slightly better class@2 and aspect@3, but class@3 and title@3 are worse, and title recovery remains very low. Qwen still does not use abstention.

### Diagnostics

Top recurring Haiku direct/no-think mistakes included:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Seer -> Mage`, `Thief -> Prince`, `Page -> Maid`, `Rogue -> Bard`, `Bard -> Prince`, `Heir -> Prince`, `Sylph -> Mage`, `Bard -> Witch` |
| Aspect | `Breath -> Hope`, `Light -> Mind`, `Hope -> Light`, `Space -> Life`, `Life -> Hope`, `Blood -> Light`, `Doom -> Mind`, `Heart -> Mind` |

Haiku changed the error shape but did not remove the major sink behavior. Aspect prediction still over-predicted `Mind`, `Light`, and `Hope`, while under-predicting `Breath`, `Time`, `Space`, `Blood`, and `Void`.

### Outcome

The final model comparison does not justify replacing the existing promoted artifacts or generating another full dataset. Claude Haiku is a cleaner anonymous-description generator than local Qwen, but the downstream classifier did not improve on the metrics that matter most. The local-Qwen baseline still has better raw and vote aspect accuracy, and both models remain weak on full-title recovery.

Do not promote the Haiku baseline artifacts. Do not run MIPRO based on the Haiku result. Further improvement likely requires a different modeling strategy, label representation, or source of supervision rather than more prompt-only anonymous-description generation.


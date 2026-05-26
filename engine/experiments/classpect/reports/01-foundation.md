# Classpect Classifier Experiment Report

## Goal

Build a classpecting classifier that predicts Homestuck class and aspect from personality evidence, while avoiding direct identity leakage from character names or canonical metadata.

The ground truth labels are the 32 canonical examples in `experiments/classpect/training_examples.py`: the 8 kids, 12 Alternian trolls, and 12 dancestors.

## Data Policy

Generated descriptions are intended to be anonymous personality summaries. They should not include character names, aliases, handles, initials, typing quirks, species, blood color, caste, gender, dream moon, lusus, planet, weapon, powers, canonical class/aspect/title, or direct plot identifiers.

Generated artifacts live under `experiments/classpect/generated/`, which is ignored by git.

Current generated description dataset:

| Item | Count |
| --- | ---: |
| Generated records | 160 |
| Valid records after filtering | 147 |
| Characters represented | 32 |
| Minimum valid variants per character | 2 |
| Maximum valid variants per character | 5 |
| Invalid records filtered out | 13 |
## Attempt 1: Decision Tree Over Extracted Features

### Method

The first approach generated personality descriptions, extracted a fixed feature vector, then trained transparent `scikit-learn` decision trees.

Feature names are defined in `experiments/classpect/decision_tree.py`:

```text
personal_agency, social_initiative, emotional_openness, self_presentation,
rule_orientation, risk_tolerance, pragmatism, idealism, relational_focus,
identity_focus, knowledge_focus, chaos_tolerance, responsibility_drive,
secrecy, conflict_directness, change_orientation
```

The scripts were:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.extract_features
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_decision_tree
```

### Results

Default tree depth is `max_depth=5`.

| Target | Training Accuracy | Leave-One-Out Accuracy |
| --- | ---: | ---: |
| Class | 0.812 | 0.094 |
| Aspect | 0.719 | 0.062 |

Unlimited depth used `--max-depth 0`.

| Target | Training Accuracy | Leave-One-Out Accuracy |
| --- | ---: | ---: |
| Class | 1.000 | 0.062 |
| Aspect | 1.000 | 0.031 |

### Outcome

The decision tree was useful for transparency, but it badly overfit. The leave-one-out scores were near chance despite high training accuracy, especially with unlimited depth.

This approach was not selected as the main classifier.

## Attempt 2: DSPy Joint Title Program

### Method

The first DSPy approach used one `ClasspectProgram` that predicted both class and aspect together. It optimized against `title_metric`, where exact title matches receive full credit and one-correct-of-two receives partial credit.

This path is still available with:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --joint
```

Saved joint artifacts include:

```text
experiments/classpect/generated/dspy_classpect_program.json
experiments/classpect/generated/dspy_classpect_program_fold_0.json
```

### Problems Found

This path was abandoned for two reasons.

First, the joint objective was noisy. Class and aspect errors were entangled, so optimization feedback was less useful than optimizing class and aspect separately.

Second, the saved joint artifacts were produced from an earlier description style that leaked character names. Grep confirmed names such as `Rose Lalonde`, `Dave Strider`, `Jade Harley`, `Karkat Vantas`, and `Meenah Peixes` inside the joint program artifacts.

### Outcome

The joint title program should not be used as a prompt source. Its saved artifacts are stale and contaminated by character-name leakage.

## Attempt 3: DSPy Separate Class And Aspect Programs

### Method

The current DSPy approach trains separate modules:

| Program | Prediction |
| --- | --- |
| `ClassProgram` | Class only |
| `AspectProgram` | Aspect only |
| `SeparateClasspectProgram` | Wraps both predictions |

The default optimizer is DSPy `MIPROv2`. It defaults to the
local model (`LOCAL_LLM_API_BASE`), model `openai/Qwen3.6-35B-A3B-MTP-GGUF`.

The standard command is:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --num-threads 8 --save
```

The important evaluation change was grouped cross-validation by character. All variants of a held-out character stay in the dev fold, preventing variant leakage between train and dev.

### Earlier 3-Variant Run

An earlier separate class/aspect DSPy run used 3 variants per character.

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.46875 |
| Mean dev aspect accuracy | 0.43750 |
| Mean dev title accuracy | 0.28125 |

This was before the stricter 5-variant no-name dataset and before the stale-artifact fingerprint hardening.

### 5-Variant No-Name Run Before Stale Correction

The first 5-variant no-name aggregate looked like this:

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.30362 |
| Mean dev aspect accuracy | 0.40784 |
| Mean dev title accuracy | 0.13443 |
| Mean dev vote class accuracy | 0.31250 |
| Mean dev vote aspect accuracy | 0.40625 |
| Mean dev vote title accuracy | 0.15625 |

This aggregate was later found to include stale fold artifacts. Fold 3 had reused older optimized programs, and the saved program contained character-name leakage. This made the aggregate untrustworthy.

### Fingerprint And Resume Hardening

To prevent stale artifact reuse, the pipeline now records dataset fingerprints and fold metadata.

The report model now includes:

```text
dataset_fingerprint
folds_total
train_evaluated
```

The saved-program metadata file is:

```text
experiments/classpect/generated/dspy_fold_<n>_metadata.json
```

Resume now ignores stale reports or saved programs when the fingerprint, total fold count, train size, or dev size does not match.

Tests were added for dataset fingerprint changes and fold metadata round-tripping.

### Current 5-Variant No-Name Results

Fold 3 was regenerated with no character-name leakage in the saved separate class/aspect programs. Because Anthropic usage limits interrupted full train evaluation, fold 3 was completed with `--skip-train-eval`, evaluating only held-out dev examples from the saved programs.

Current aggregate report:

```text
experiments/classpect/generated/dspy_classpect_report.json
```

| Metric | Value |
| --- | ---: |
| Mean train class accuracy | 0.55103 |
| Mean train aspect accuracy | 0.33931 |
| Mean train title accuracy | 0.15387 |
| Mean dev class accuracy | 0.32445 |
| Mean dev aspect accuracy | 0.39395 |
| Mean dev title accuracy | 0.12749 |
| Mean dev vote class accuracy | 0.28125 |
| Mean dev vote aspect accuracy | 0.37500 |
| Mean dev vote title accuracy | 0.12500 |

Fold-level dev metrics:

| Fold | Train Size | Dev Size | Train Evaluated | Dev Class | Dev Aspect | Dev Title | Vote Class | Vote Aspect | Vote Title |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 112 | 35 | yes | 0.257 | 0.514 | 0.086 | 0.250 | 0.375 | 0.000 |
| 1 | 110 | 37 | yes | 0.459 | 0.324 | 0.189 | 0.500 | 0.375 | 0.250 |
| 2 | 108 | 39 | yes | 0.359 | 0.487 | 0.179 | 0.375 | 0.500 | 0.250 |
| 3 | 111 | 36 | no | 0.222 | 0.250 | 0.056 | 0.000 | 0.250 | 0.000 |

### Prompt Selection

The best current saved prompts are separate class and aspect prompts, not the joint title prompt.

Best class fold:

```text
experiments/classpect/generated/dspy_class_program_fold_1.json
```

Best aspect fold:

```text
experiments/classpect/generated/dspy_aspect_program_fold_2.json
```

These were promoted to the default filenames:

```text
experiments/classpect/generated/dspy_class_program.json
experiments/classpect/generated/dspy_aspect_program.json
```

The promoted default class/aspect program files were checked for obvious character-name leakage and did not match the sampled character-name grep.

## Attempt 5: Final All-Data Bootstrap Run

The trainer supports a final separate class/aspect training run on all valid anonymous descriptions, using cross-validation only as the performance estimate:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --final --save
```

This writes `dspy_final_class_program.json`, `dspy_final_aspect_program.json`, and `dspy_final_classpect_report.json`. The final report only measures train-set fit; the grouped CV report remains the generalization estimate.

If final training is interrupted after saving programs, resume with `--final --resume --save`. Add `--skip-train-eval` when only the saved final artifacts are needed and the train-fit pass is too expensive.

The current final all-data bootstrap artifacts were generated from all 147 valid anonymous descriptions. Their train-fit metrics are:

| Metric | Value |
| --- | ---: |
| Train class accuracy | 0.286 |
| Train aspect accuracy | 0.252 |
| Train title accuracy | 0.075 |
| Train vote class accuracy | 0.281 |
| Train vote aspect accuracy | 0.250 |
| Train vote title accuracy | 0.094 |

One row recorded a prediction error because the model returned `Justice`, which is not a valid aspect. Evaluation now records malformed or invalid per-example predictions as wrong rows with `prediction_error` instead of aborting the full report.

The low train-fit score is informative. Even when trained and evaluated on all valid descriptions, the bootstrap prompt often fails to recover the canon labels. That points to model/prompt/demonstration limits and label ambiguity, not just held-out generalization failure.

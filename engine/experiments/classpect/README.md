# Classpect Classifier Experiment

*Note: This sub-project, including experiment reports, most code, and documentation including the README, was largely AI-generated.*

Build a classpecting classifier that predicts Homestuck class and aspect from anonymous personality summaries, while avoiding direct identity leakage from character names or canonical metadata.

- **Ground truth**: 32 canon-labeled characters (8 kids, 12 Alternian trolls, 12 dancestors) from `training_examples.py`
- **Data policy**: Generated descriptions must not include names, aliases, species, blood color, caste, gender, dream moon, lusus, planet, weapons, powers, canonical class/aspect/title, or direct plot identifiers.
- **Generated artifacts** live under `generated/` (gitignored)
- **API key and base URL** are read from `LOCAL_LLM_API_KEY` and `LOCAL_LLM_API_BASE` environment variables

Run commands from `engine/` with `PYTHONPATH=.` set.

## Experiment Summary

The project ran 16 attempts across 6 thematic phases. Full reports are in `reports/`.

### Phase 1: Foundation (`reports/01-foundation.md`)

| Attempt | Approach | Result |
| --- | --- | --- |
| 1 | Decision tree over extracted 16-dim feature vectors | Near-chance leave-one-out (class 0.094, aspect 0.062) |
| 2 | DSPy joint title program (class+aspect together) | Abandoned — joint objective noisy, stale artifacts leaked names |
| 3 | DSPy separate class & aspect programs with grouped CV | **Promoted as best artifacts**. Class 0.324, aspect 0.394, title 0.127 |
| 5 | Final all-data bootstrap on all 147 valid descriptions | Low train fit (class 0.286, aspect 0.252) — label ambiguity limits even train recovery |

### Phase 2: Prompt Engineering (`reports/02-prompt-engineering.md`)

| Attempt | Approach | Result |
| --- | --- | --- |
| 4 | Profile-guided rubrics (LLM-generated label mini-descriptions) | Worse than baseline (class 0.167, aspect 0.131) |
| 6 | Qwen no-think / direct predictor stability | Direct/no-think eliminated parse errors but underperformed promoted artifacts |
| 7 | Constrained direct/no-think labels (exact label only, no explanation) | Vote aspect improved from 0.219 to 0.281, still far below promoted baseline |
| 14 | Clean nested MIPRO with inner validation split | Improved constrained class (0.191) but regressed aspect (0.187) |

### Phase 3: Description Dataset Iteration (`reports/03-description-iteration.md`)

| Attempt | Approach | Result |
| --- | --- | --- |
| 8 | Contrastive v2 descriptions guided by confusion diagnostics | Better raw class (0.176), collapsed aspect (0.058) |
| 9 | Aspect-balanced v3 descriptions | Failed — aspect fell further (0.029), spread errors to Doom/Life/Rage |
| 11 | Invalid-generation analysis | Retry hardening added rejection-specific regeneration |
| 12 | Retry-hardened smoke dataset | 0% invalid rate but no accuracy improvement |
| 13 | Claude Haiku vs Qwen generation comparison | Haiku cleaner generator (0% invalid) but no downstream improvement |

### Phase 4: Deep Diagnostics (`reports/04-diagnostics.md`)

| Attempt | Approach | Result |
| --- | --- | --- |
| 10 | Ranked top-k and abstention evaluation | Class@3 reached 0.408 on v1, but aspect@3 stayed at 0.293 — model often can't rank true aspect in top 3 |
| 15 | Pairwise confusion audit (2-label forced choice) | Overall recovery 0.582; aspect pairwise 0.558 — below deployable threshold |

### Phase 5: Theory Induction (`reports/05-theory-induction.md`)

| Attempt | Approach | Result |
| --- | --- | --- |
| 16 | Fold-local contrastive theory induction (personality-only) | Class 0.197, aspect 0.170 — worse than direct classifiers |
| 16 | Personality + abstract-role theory induction | Collapsed — aspect 0.038, title 0.000 |

### Phase 6: Conclusion (`reports/06-conclusion.md`)

The promoted separate class/aspect DSPy programs remain the best available artifacts despite their modest accuracy. The key finding is that LLM-generated anonymous personality summaries lack sufficient discriminative signal for aspect and full-title recovery. Future work should move toward auditable representations (hand-curated evidence, binary questions, human-authored features).

## Current Best Artifacts

```
generated/dspy_class_program.json     — Best class predictor (fold 1)
generated/dspy_aspect_program.json    — Best aspect predictor (fold 2)
```

## Pipeline Scripts

Scripts are organized by theme under `scripts/`:

### Foundation (`scripts/foundation/`)

| Script | Purpose |
| --- | --- |
| `train_dspy_classpect.py` | Main DSPy trainer: grouped CV, MIPRO/bootstrap, separate or joint, final all-data |
| `train_decision_tree.py` | scikit-learn decision tree with leave-one-out CV |
| `extract_features.py` | Extract 16-dim feature vectors from descriptions |
| `diagnose_dspy_report.py` | Confusion/bias diagnostics from saved reports |

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --predictor direct --qwen-no-think --optimizer bootstrap --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_dspy_classpect --final --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.diagnose_dspy_report --run-name direct_no_think --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.extract_features
PYTHONPATH=. uv run python -m experiments.classpect.scripts.foundation.train_decision_tree
```

### Prompt Engineering (`scripts/prompt_engineering/`)

| Script | Purpose |
| --- | --- |
| `generate_label_profiles.py` | Generate label mini-descriptions per fold for rubric |
| `train_profile_guided_dspy_classpect.py` | Train classifier with profile rubric |
| `evaluate_profile_similarity.py` | Deterministic TF-IDF profile similarity (no LLM) |
| `evaluate_description_similarity.py` | Deterministic description nearest-neighbor (no LLM) |

```bash
for fold in 0 1 2 3; do
  PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.generate_label_profiles --fold "$fold" --num-threads 4
done
PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.train_profile_guided_dspy_classpect --num-threads 8 --save
PYTHONPATH=. uv run python -m experiments.classpect.scripts.prompt_engineering.evaluate_profile_similarity
```

### Generation (`scripts/generation/`)

| Script | Purpose |
| --- | --- |
| `generate_descriptions.py` | Generate anonymous personality descriptions (Anthropic or OpenAI-compatible) |
| `summarize_descriptions.py` | Dataset stats and fingerprint |
| `analyze_invalid_descriptions.py` | Diagnose forbidden-term leakage patterns |

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --variants 5 --run-name contrast_v2 --concurrency 2
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.summarize_descriptions
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.analyze_invalid_descriptions --descriptions-path generated/descriptions.json
```

### Evaluation (`scripts/evaluation/`)

| Script | Purpose |
| --- | --- |
| `evaluate_ranked_classpect.py` | Top-k ranked class/aspect prediction with abstention |
| `evaluate_pairwise_classpect.py` | Forced-choice confusion-pair audit |

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.evaluation.evaluate_ranked_classpect --run-name ranked_v1
PYTHONPATH=. uv run python -m experiments.classpect.scripts.evaluation.evaluate_pairwise_classpect --run-name pairwise_v1_confusions
```

### Theory Induction (`scripts/theory_induction/`)

| Script | Purpose |
| --- | --- |
| `evaluate_theory_induction.py` | Fold-local theory induction + scored prediction |

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.theory_induction.evaluate_theory_induction --variant personality_only
```

## Key Metrics (Promoted Artifacts)

| Metric | Value |
| --- | ---: |
| Mean dev class accuracy | 0.324 |
| Mean dev aspect accuracy | 0.394 |
| Mean dev title accuracy | 0.127 |
| Mean dev vote class accuracy | 0.281 |
| Mean dev vote aspect accuracy | 0.375 |
| Mean dev vote title accuracy | 0.125 |

## Key Constraints

- **Grouped CV**: All variants of a held-out character stay in the dev fold. No variant leakage.
- **Fingerprinting**: Dataset hashes prevent stale-artifact reuse across description versions.
- **Prediction error tracking**: Invalid labels (e.g., `Justice`) are recorded as errors, not silently dropped.
- **No non-canonical examples**: The experiment stays grounded in the 32 canon-labeled characters.

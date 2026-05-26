# Classpect Experiment Report — Part 5: Theory Induction

## Attempt 16: Fold-Local Contrastive Theory Induction

### Premise

Do not import a fixed external classpect rubric as if it were universally valid. Existing classpect interpretations are numerous, overlapping, often inspired by each other, and may drift from the original source material. Consolidating outside analyst theories would add a new subjective target rather than solving the current generalization problem.

Instead, treat classpect theory itself as the object being induced and tested. The experiment should ask whether a local operational interpretation of each class and aspect can be inferred from the canon-labeled examples and then generalize to held-out characters.

The target transformation is:

```text
canon-labeled train characters -> fold-local label interpretations -> held-out character prediction
```

This is different from direct classification because the model must first make its implied theory explicit, then use that theory as an intermediate artifact.

### Hypothesis

The previous classifier path may have failed because it asked the model to map anonymous descriptions directly to labels without first constructing a local explanation of what each label means in this specific canon-labeled dataset.

Attempt 16 tests whether fold-local, canon-grounded, contrastive label interpretations improve held-out generalization without relying on a universal external theory.

### Method

For each grouped CV fold:

1. Hold out dev characters completely.
2. Use only train characters and their canon labels to induce class and aspect theories.
3. For each class and aspect label, generate a fold-local label brief.
4. Each label brief should include common evidence among train examples with that label, contrastive distinctions against common confusions, and uncertainty notes when train support is sparse.
5. Predict held-out characters by comparing their anonymous dossier to every fold-local label brief.
6. Save both the induced theories and the scored predictions for inspection.
7. Report class, aspect, title, ranked top-k, margins, abstentions, and errors.

The theory induction prompt must explicitly avoid external analyst theory unless a later exploratory variant marks that source separately. The first run should be canon-labeled-data-only.

### Input Variants

Run the first variant before adding more inputs:

| Variant | Input | Purpose |
| --- | --- | --- |
| `personality_only` | Existing anonymous v1 personality descriptions | Tests whether explicit fold-local theory induction improves the current representation. |
| `personality_plus_abstract_role` | Personality plus abstract narrative-function evidence | Tests whether prior inputs were too lossy for classpecting. Run only after `personality_only` is inspected. |

The optional abstract-role variant must remain anonymous. It may use statements such as `often catalyzes group movement while avoiding direct responsibility`, but must not include names, aliases, handles, initials, species, blood color, caste, gender, dream moon, lusus, planet, weapons, powers, canonical title/class/aspect, direct plot identifiers, or role labels such as `the protagonist`.

### Theory Artifact Shape

Each induced label brief should be a first-class artifact, not hidden inside a prompt. The intended schema is:

```text
target: class | aspect
label: valid class/aspect label
source_character_count: number of train characters with this label
source_description_count: number of train descriptions used
common_evidence: concise bullet list
contrastive_evidence: concise bullet list, especially against recurring sink/confusion labels
uncertainty_notes: concise notes about sparse support or ambiguous evidence
```

Prediction rows should include:

```text
character_id
variant
expected_class
expected_aspect
ranked_classes with scores
ranked_aspects with scores
class_margin
aspect_margin
abstained flags
raw theory-scoring output
prediction_error
```

### Expected Files

Initial personality-only artifacts:

```text
experiments/classpect/generated/fold_theories_personality_only_fold_0.json
experiments/classpect/generated/fold_theories_personality_only_fold_1.json
experiments/classpect/generated/fold_theories_personality_only_fold_2.json
experiments/classpect/generated/fold_theories_personality_only_fold_3.json
experiments/classpect/generated/theory_induction_report_personality_only.json
experiments/classpect/generated/theory_induction_diagnostics_personality_only.json
```

Optional abstract-role artifacts, if the first run justifies continuing:

```text
experiments/classpect/generated/fold_theories_personality_plus_role_fold_0.json
experiments/classpect/generated/theory_induction_report_personality_plus_role.json
experiments/classpect/generated/theory_induction_diagnostics_personality_plus_role.json
```

### Evaluation Gates

Continue if at least one of these improves clearly:

1. Aspect dev accuracy exceeds the promoted baseline of `0.394`.
2. Aspect confusion/pairwise recovery reaches roughly `0.65` or better.
3. Ranked aspect or full-title top-k improves materially with useful margins or abstentions.

Stop or redesign if:

1. Fold-local theories collapse into generic personality typology language.
2. Theories leak canon identifiers or held-out character evidence.
3. Aspect remains near or below prior direct classifiers.
4. Abstract role evidence does not materially improve aspect or title recovery.

### Implementation Plan

1. Add theory-induction data models such as `InducedLabelTheory`, `FoldTheorySet`, `TheoryScoredPrediction`, and `TheoryInductionReport`.
2. Add `experiments.classpect.scripts.theory_induction.evaluate_theory_induction.py`.
3. Start with `personality_only` using existing v1 descriptions and local Qwen.
4. Enforce fold isolation: no held-out character descriptions may appear in theory induction inputs.
5. Add tests for schema parsing, fold isolation, no held-out leakage, ranked scoring, margins, abstentions, and report metrics.
6. Run focused tests before any full evaluation.
7. Run the `personality_only` evaluation and inspect induced theories before deciding on `personality_plus_abstract_role` dossiers.

### Personality-Only Run

The first implementation added a fold-local theory induction evaluator and ran the `personality_only` variant against the existing v1 anonymous descriptions. The evaluator saves one theory file per fold, scores held-out descriptions against fold-local class/aspect theories, and writes a report plus diagnostics.

The run used local Qwen:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.theory_induction.evaluate_theory_induction --variant personality_only --resume --max-theory-tokens 1024 --max-score-tokens 2048 --request-timeout 300
```

The initial scoring pass produced a `0.143` prediction error rate from truncated score JSON. The scorer prompt was tightened, malformed theory JSON parsing was hardened, and failed rows were retried with a larger scoring budget:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.theory_induction.evaluate_theory_induction --variant personality_only --resume --retry-errors --max-theory-tokens 1024 --max-score-tokens 4096 --request-timeout 300
```

Final artifacts:

```text
experiments/classpect/generated/fold_theories_personality_only_fold_0.json
experiments/classpect/generated/fold_theories_personality_only_fold_1.json
experiments/classpect/generated/fold_theories_personality_only_fold_2.json
experiments/classpect/generated/fold_theories_personality_only_fold_3.json
experiments/classpect/generated/theory_induction_report_personality_only.json
experiments/classpect/generated/theory_induction_diagnostics_personality_only.json
```

Dataset fingerprint:

```text
63e997f24372b57ef72df61470bc56626d248979d22324f79203638412a662a0
```

### Personality-Only Results

| Metric | Value |
| --- | ---: |
| Rows | 147 |
| Class accuracy | 0.197 |
| Aspect accuracy | 0.170 |
| Title accuracy | 0.020 |
| Class@3 | 0.415 |
| Aspect@3 | 0.367 |
| Title@3 | 0.177 |
| Class abstention | 0.007 |
| Aspect abstention | 0.007 |
| Prediction error | 0.000 |

Fold-level results:

| Fold | Rows | Class | Aspect | Title | Class@3 | Aspect@3 | Title@3 | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35 | 0.143 | 0.086 | 0.000 | 0.314 | 0.400 | 0.114 | 0.000 |
| 1 | 37 | 0.162 | 0.162 | 0.000 | 0.514 | 0.297 | 0.270 | 0.000 |
| 2 | 39 | 0.282 | 0.282 | 0.077 | 0.513 | 0.410 | 0.231 | 0.000 |
| 3 | 36 | 0.194 | 0.139 | 0.000 | 0.306 | 0.361 | 0.083 | 0.000 |

Compared with the promoted separate class/aspect artifacts, personality-only theory induction is worse on every primary metric:

| Run | Class | Aspect | Title | Class@3 | Aspect@3 | Title@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted separate class/aspect artifacts | 0.324 | 0.394 | 0.127 | not reported | not reported | not reported |
| personality-only fold-local theory induction | 0.197 | 0.170 | 0.020 | 0.415 | 0.367 | 0.177 |

### Personality-Only Diagnostics

Top confusions:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Witch -> Maid`, `Bard -> Page`, `Thief -> Prince`, `Rogue -> Heir`, `Seer -> Mage`, `Heir -> Rogue`, `Prince -> Knight`, `Heir -> Bard` |
| Aspect | `Space -> Heart`, `Hope -> Blood`, `Breath -> Hope`, `Doom -> Blood`, `Heart -> Hope`, `Life -> Hope`, `Light -> Mind`, `Space -> Breath` |

Largest prediction-bias deltas:

| Target | Largest Over-Predictions | Largest Under-Predictions |
| --- | --- | --- |
| Class | `Page=+11`, `Knight=+7`, `Maid=+7`, `Mage=+5` | `Witch=-10`, `Thief=-9`, `Rogue=-8`, `Heir=-5` |
| Aspect | `Hope=+12`, `Blood=+11`, `Heart=+9`, `Breath=+7` | `Space=-13`, `Time=-10`, `Void=-9`, `Light=-7` |

The theory artifacts did not leak character names in the saved fold theory files, but several raw/generated theory notes still drifted into broad narrative-role language such as `protagonist`, `supporting character`, and `explicit plot or title cues`. That is useful diagnostically: the model appears to want narrative-functional evidence, but the personality-only representation does not provide it cleanly.

### Personality-Only Outcome

The `personality_only` theory-induction variant fails the continuation gate. It does not improve over the promoted artifacts, does not improve aspect recovery, and performs worse than the prior direct classifiers. The intermediate theory artifacts are inspectable, but making the theory explicit did not make personality-only descriptions sufficient.

Do not promote these artifacts. Do not run MIPRO on this representation. Do not build a larger tournament on top of it.

If continuing Attempt 16, the next meaningful change is not more optimization; it is the optional `personality_plus_abstract_role` input variant, treated as a representation test. That variant should be designed carefully because the personality-only theory artifacts already show pressure toward narrative-role concepts, and those concepts must be represented without direct plot identifiers or label leakage.

### Abstract-Role Dossier Run

The optional representation test generated new anonymous dossiers that include personality plus abstract narrative-function evidence. The generator was extended with a `personality-plus-abstract-role` prompt style and stricter validation against explicit role labels such as `protagonist`, `antagonist`, `main character`, `supporting character`, `side character`, `villain`, and `hero`.

Generation command:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.generation.generate_descriptions --backend openai-compatible --prompt-style personality-plus-abstract-role --run-name personality_plus_abstract_role --variants 5 --concurrency 4 --max-retries 3 --max-tokens 2048 --request-timeout 300 --resume --retry-invalid
```

Two follow-up retry-invalid passes were needed for leaked metadata terms. Final dataset summary:

| Metric | Value |
| --- | ---: |
| Generated records | 160 |
| Valid records | 160 |
| Invalid records | 0 |
| Characters represented | 32 |
| Minimum valid variants per character | 5 |
| Maximum valid variants per character | 5 |

Dataset path and fingerprint:

```text
experiments/classpect/generated/descriptions_personality_plus_abstract_role.json
1d935aab2a0d88c0f68ea3549d9dbd75c9a2f45b6d73ca84dce9d33cbd95e74a
```

Theory-induction evaluation command:

```bash
PYTHONPATH=. uv run python -m experiments.classpect.scripts.theory_induction.evaluate_theory_induction --descriptions-path experiments/classpect/generated/descriptions_personality_plus_abstract_role.json --variant personality_plus_abstract_role --resume --max-theory-tokens 1024 --max-score-tokens 4096 --request-timeout 300
```

Generated artifacts:

```text
experiments/classpect/generated/fold_theories_personality_plus_abstract_role_fold_0.json
experiments/classpect/generated/fold_theories_personality_plus_abstract_role_fold_1.json
experiments/classpect/generated/fold_theories_personality_plus_abstract_role_fold_2.json
experiments/classpect/generated/fold_theories_personality_plus_abstract_role_fold_3.json
experiments/classpect/generated/theory_induction_report_personality_plus_abstract_role.json
experiments/classpect/generated/theory_induction_diagnostics_personality_plus_abstract_role.json
```

### Abstract-Role Results

| Metric | Value |
| --- | ---: |
| Rows | 160 |
| Class accuracy | 0.125 |
| Aspect accuracy | 0.038 |
| Title accuracy | 0.000 |
| Class@3 | 0.275 |
| Aspect@3 | 0.175 |
| Title@3 | 0.056 |
| Class abstention | 0.000 |
| Aspect abstention | 0.000 |
| Prediction error | 0.000 |

Fold-level results:

| Fold | Rows | Class | Aspect | Title | Class@3 | Aspect@3 | Title@3 | Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 40 | 0.025 | 0.050 | 0.000 | 0.050 | 0.175 | 0.000 | 0.000 |
| 1 | 40 | 0.150 | 0.050 | 0.000 | 0.325 | 0.300 | 0.100 | 0.000 |
| 2 | 40 | 0.175 | 0.000 | 0.000 | 0.450 | 0.150 | 0.125 | 0.000 |
| 3 | 40 | 0.150 | 0.050 | 0.000 | 0.275 | 0.075 | 0.000 | 0.000 |

Compared across Attempt 16 variants:

| Run | Class | Aspect | Title | Class@3 | Aspect@3 | Title@3 | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| personality-only theory induction | 0.197 | 0.170 | 0.020 | 0.415 | 0.367 | 0.177 | 0.000 |
| personality-plus-abstract-role theory induction | 0.125 | 0.038 | 0.000 | 0.275 | 0.175 | 0.056 | 0.000 |
| promoted separate class/aspect artifacts | 0.324 | 0.394 | 0.127 | not reported | not reported | not reported | historical clean CV |

### Abstract-Role Diagnostics

Top confusions:

| Target | Repeated Mistakes |
| --- | --- |
| Class | `Thief -> Bard`, `Sylph -> Mage`, `Maid -> Seer`, `Heir -> Knight`, `Page -> Maid`, `Rogue -> Page`, `Rogue -> Prince`, `Seer -> Mage`, `Witch -> Bard` |
| Aspect | `Breath -> Hope`, `Void -> Blood`, `Rage -> Hope`, `Hope -> Blood`, `Blood -> Mind`, `Heart -> Life`, `Heart -> Void`, `Life -> Rage`, `Light -> Breath`, `Time -> Rage` |

Largest prediction-bias deltas:

| Target | Largest Over-Predictions | Largest Under-Predictions |
| --- | --- | --- |
| Class | `Bard=+16`, `Mage=+15`, `Knight=+5`, `Page=+4` | `Witch=-11`, `Rogue=-10`, `Sylph=-7`, `Thief=-7`, `Seer=-5` |
| Aspect | `Blood=+11`, `Hope=+11`, `Rage=+9`, `Breath=+8` | `Time=-12`, `Life=-10`, `Light=-7`, `Space=-7` |

The generated dossier file passed the stricter forbidden-term validation with `0` invalid records. A direct grep for role-label and metadata terms in the description file found no matches outside JSON metadata keys. However, the abstract-role representation made theory induction worse rather than better. The model appears to amplify broad narrative-function archetypes into new sink labels instead of recovering the canon labels.

### Abstract-Role Outcome

The `personality_plus_abstract_role` representation test fails decisively. It underperforms both the promoted artifacts and the already-failed personality-only theory-induction run. Adding abstract narrative-function evidence in this generated form did not supply usable classpect signal; it made aspect recovery nearly collapse.

Do not promote the abstract-role dossiers, fold theories, report, or diagnostics. Do not run MIPRO or tournament ranking on this representation.

The next strategy should not be another generated-dossier variant. The evidence now points away from LLM-generated anonymous descriptions as the primary representation. If continuing, use a non-generative, explicitly auditable representation: either hand-curated canon evidence snippets, a small human-authored feature matrix, or a narrow binary-question instrument whose questions can be inspected independently of label predictions.


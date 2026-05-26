import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel

from experiments.classpect.decision_tree import GENERATED_DIR
from experiments.classpect.dspy_classpect import (
    VALID_ASPECTS,
    VALID_CLASSES,
    character_name,
)
from experiments.classpect.training_examples import TRAINING_EXAMPLES

TheoryTarget = Literal["class", "aspect"]


class InducedLabelTheory(BaseModel):
    target: TheoryTarget
    label: str
    source_character_count: int
    source_description_count: int
    common_evidence: list[str]
    contrastive_evidence: list[str]
    uncertainty_notes: list[str]
    raw_output: str = ""


class FoldTheorySet(BaseModel):
    variant: str
    fold: int
    folds_total: int
    dataset_fingerprint: str
    train_character_ids: list[int]
    dev_character_ids: list[int]
    theories: list[InducedLabelTheory]


class TheoryLabelScore(BaseModel):
    label: str
    score: float
    evidence: str = ""
    rationale: str = ""


class TheoryScoredPrediction(BaseModel):
    fold: int
    character: str
    character_id: int | None = None
    variant: int | None = None
    expected_class: str
    expected_aspect: str
    predicted_class: str | None = None
    predicted_aspect: str | None = None
    class_match: bool = False
    aspect_match: bool = False
    title_match: bool = False
    ranked_classes: list[TheoryLabelScore] = []
    ranked_aspects: list[TheoryLabelScore] = []
    class_margin: float | None = None
    aspect_margin: float | None = None
    class_abstained: bool = False
    aspect_abstained: bool = False
    raw_class_prediction: str = ""
    raw_aspect_prediction: str = ""
    prediction_error: str | None = None


class TheoryFoldSummary(BaseModel):
    fold: int
    row_count: int
    class_accuracy: float
    aspect_accuracy: float
    title_accuracy: float
    class_top_3_accuracy: float
    aspect_top_3_accuracy: float
    title_top_3_accuracy: float
    class_abstention_rate: float
    aspect_abstention_rate: float
    prediction_error_rate: float


class TheoryInductionReport(BaseModel):
    variant: str
    dataset_fingerprint: str
    folds: int
    row_count: int
    class_accuracy: float
    aspect_accuracy: float
    title_accuracy: float
    class_top_3_accuracy: float
    aspect_top_3_accuracy: float
    title_top_3_accuracy: float
    class_abstention_rate: float
    aspect_abstention_rate: float
    prediction_error_count: int
    prediction_error_rate: float
    fold_summaries: list[TheoryFoldSummary]
    rows: list[TheoryScoredPrediction]


class TheoryLabelConfusion(BaseModel):
    expected: str
    predicted: str
    count: int
    characters: list[str]


class TheoryPredictionBias(BaseModel):
    label: str
    expected_count: int
    predicted_count: int
    delta: int


class TheoryInductionDiagnostics(BaseModel):
    row_count: int
    top_class_confusions: list[TheoryLabelConfusion]
    top_aspect_confusions: list[TheoryLabelConfusion]
    class_prediction_bias: list[TheoryPredictionBias]
    aspect_prediction_bias: list[TheoryPredictionBias]


def safe_variant(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")


def fold_theories_path(variant: str, fold: int) -> Path:
    return GENERATED_DIR / f"fold_theories_{safe_variant(variant)}_fold_{fold}.json"


def theory_induction_report_path(variant: str) -> Path:
    return GENERATED_DIR / f"theory_induction_report_{safe_variant(variant)}.json"


def theory_induction_diagnostics_path(variant: str) -> Path:
    return GENERATED_DIR / f"theory_induction_diagnostics_{safe_variant(variant)}.json"


def labels_for_theory_target(target: TheoryTarget) -> tuple[str, ...]:
    return VALID_CLASSES if target == "class" else VALID_ASPECTS


def expected_label(example: dspy.Example, target: TheoryTarget) -> str:
    return example.title_class if target == "class" else example.title_aspect


def character_ids(examples: Sequence[dspy.Example]) -> list[int]:
    return sorted(
        {
            int(example.character_id)
            for example in examples
            if getattr(example, "character_id", None) is not None
        }
    )


def one_variant_per_character(examples: Sequence[dspy.Example]) -> list[dspy.Example]:
    by_character: dict[int, dspy.Example] = {}
    for example in sorted(
        examples,
        key=lambda item: (
            int(getattr(item, "character_id", 10_000)),
            int(getattr(item, "variant", 10_000)),
        ),
    ):
        if getattr(example, "character_id", None) is None:
            continue
        by_character.setdefault(int(example.character_id), example)
    return list(by_character.values())


def source_examples_for_label(
    examples: Sequence[dspy.Example], target: TheoryTarget, label: str
) -> list[dspy.Example]:
    return [
        example
        for example in one_variant_per_character(examples)
        if expected_label(example, target) == label
    ]


def contrast_examples_for_label(
    examples: Sequence[dspy.Example], target: TheoryTarget, label: str
) -> list[dspy.Example]:
    selected_by_label: dict[str, dspy.Example] = {}
    for example in one_variant_per_character(examples):
        other_label = expected_label(example, target)
        if other_label == label:
            continue
        selected_by_label.setdefault(other_label, example)
    return [selected_by_label[other_label] for other_label in sorted(selected_by_label)]


def format_theory_examples(examples: Sequence[dspy.Example], target: TheoryTarget) -> str:
    lines = []
    for index, example in enumerate(examples, start=1):
        lines.append(
            f"Example {index}\n"
            f"Known {target}: {expected_label(example, target)}\n"
            f"Description:\n{example.personality_description}"
        )
    return "\n\n".join(lines) if lines else "No examples available."


def theory_context(theories: Sequence[InducedLabelTheory], target: TheoryTarget) -> str:
    sections = []
    for theory in sorted(
        [theory for theory in theories if theory.target == target], key=lambda item: item.label
    ):
        sections.append(
            f"{theory.label}\n"
            "Common evidence:\n"
            + "\n".join(f"- {item}" for item in theory.common_evidence)
            + "\nContrastive evidence:\n"
            + "\n".join(f"- {item}" for item in theory.contrastive_evidence)
            + "\nUncertainty notes:\n"
            + "\n".join(f"- {item}" for item in theory.uncertainty_notes)
        )
    return "\n\n".join(sections)


def extract_json_object(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output: {raw!r}")
        data = json.loads(match.group(0))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
    return data


def string_list(value: Any, *, fallback: str, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return (cleaned[:limit] if limit is not None else cleaned) or [fallback]
    if isinstance(value, str) and value.strip():
        lines = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            for line in value.splitlines()
        ]
        cleaned = [line for line in lines if line] or [value.strip()]
        return cleaned[:limit] if limit is not None else cleaned
    return [fallback]


def salvage_json_string_list(raw: str, key: str, *, limit: int = 3) -> list[str]:
    key_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', raw)
    if not key_match:
        return []
    start = key_match.end()
    next_key = re.search(r'\n\s*"[a-zA-Z_]+"\s*:', raw[start:])
    end = start + next_key.start() if next_key else len(raw)
    segment = raw[start:end]
    values = []
    for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', segment):
        try:
            value = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            value = match.group(1)
        value = str(value).strip()
        if value:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def parse_induced_theory(
    raw: str,
    *,
    target: TheoryTarget,
    label: str,
    source_character_count: int,
    source_description_count: int,
) -> InducedLabelTheory:
    try:
        data = extract_json_object(raw)
    except ValueError:
        data = {
            "common_evidence": salvage_json_string_list(raw, "common_evidence"),
            "contrastive_evidence": salvage_json_string_list(
                raw, "contrastive_evidence"
            ),
            "uncertainty_notes": salvage_json_string_list(raw, "uncertainty_notes"),
        }
    return InducedLabelTheory(
        target=target,
        label=label,
        source_character_count=source_character_count,
        source_description_count=source_description_count,
        common_evidence=string_list(
            data.get("common_evidence"), fallback="No common evidence identified.", limit=3
        ),
        contrastive_evidence=string_list(
            data.get("contrastive_evidence"),
            fallback="No contrastive evidence identified.",
            limit=3,
        ),
        uncertainty_notes=string_list(
            data.get("uncertainty_notes"),
            fallback="No uncertainty notes provided.",
            limit=3,
        ),
        raw_output=raw,
    )


def normalize_label(value: Any, choices: Sequence[str]) -> str | None:
    text = str(value or "").strip()
    for choice in choices:
        if text.casefold() == choice.casefold():
            return choice
    return None


def parse_theory_scores(raw: str, choices: Sequence[str]) -> tuple[list[TheoryLabelScore], bool]:
    data = extract_json_object(raw)
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError("Scoring output must include a scores list")

    scores_by_label: dict[str, TheoryLabelScore] = {}
    for item in raw_scores:
        if not isinstance(item, Mapping):
            continue
        label = normalize_label(item.get("label"), choices)
        if label is None:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        scores_by_label[label] = TheoryLabelScore(
            label=label,
            score=max(0.0, min(5.0, score)),
            evidence=str(item.get("evidence") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
        )

    if not scores_by_label:
        raise ValueError("Scoring output did not include any valid labels")
    ranked = sorted(scores_by_label.values(), key=lambda item: (-item.score, item.label))
    abstained = bool(data.get("abstain", False))
    return ranked, abstained


def prediction_margin(ranked: Sequence[TheoryLabelScore]) -> float | None:
    if not ranked:
        return None
    if len(ranked) == 1:
        return ranked[0].score
    return ranked[0].score - ranked[1].score


def ranked_hit(ranked: Sequence[TheoryLabelScore], expected: str, k: int) -> bool:
    return expected in [score.label for score in ranked[:k]]


def build_theory_prediction(
    *,
    fold: int,
    example: dspy.Example,
    raw_class_prediction: str,
    raw_aspect_prediction: str,
    class_error: str | None = None,
    aspect_error: str | None = None,
) -> TheoryScoredPrediction:
    prediction_error_parts = []
    ranked_classes: list[TheoryLabelScore] = []
    ranked_aspects: list[TheoryLabelScore] = []
    class_abstained = False
    aspect_abstained = False
    if class_error is not None:
        prediction_error_parts.append(class_error)
    else:
        try:
            ranked_classes, class_abstained = parse_theory_scores(
                raw_class_prediction, VALID_CLASSES
            )
        except ValueError as exc:
            prediction_error_parts.append(f"class: {exc}")
    if aspect_error is not None:
        prediction_error_parts.append(aspect_error)
    else:
        try:
            ranked_aspects, aspect_abstained = parse_theory_scores(
                raw_aspect_prediction, VALID_ASPECTS
            )
        except ValueError as exc:
            prediction_error_parts.append(f"aspect: {exc}")

    predicted_class = ranked_classes[0].label if ranked_classes and not class_abstained else None
    predicted_aspect = ranked_aspects[0].label if ranked_aspects and not aspect_abstained else None
    class_match = predicted_class == example.title_class
    aspect_match = predicted_aspect == example.title_aspect
    return TheoryScoredPrediction(
        fold=fold,
        character=character_name(example),
        character_id=getattr(example, "character_id", None),
        variant=getattr(example, "variant", None),
        expected_class=example.title_class,
        expected_aspect=example.title_aspect,
        predicted_class=predicted_class,
        predicted_aspect=predicted_aspect,
        class_match=class_match,
        aspect_match=aspect_match,
        title_match=class_match and aspect_match,
        ranked_classes=ranked_classes,
        ranked_aspects=ranked_aspects,
        class_margin=prediction_margin(ranked_classes),
        aspect_margin=prediction_margin(ranked_aspects),
        class_abstained=class_abstained,
        aspect_abstained=aspect_abstained,
        raw_class_prediction=raw_class_prediction,
        raw_aspect_prediction=raw_aspect_prediction,
        prediction_error="; ".join(prediction_error_parts) if prediction_error_parts else None,
    )


def summarize_rows(rows: Sequence[TheoryScoredPrediction], *, fold: int) -> TheoryFoldSummary:
    count = len(rows) or 1
    prediction_error_count = sum(row.prediction_error is not None for row in rows)
    return TheoryFoldSummary(
        fold=fold,
        row_count=len(rows),
        class_accuracy=sum(row.class_match for row in rows) / count,
        aspect_accuracy=sum(row.aspect_match for row in rows) / count,
        title_accuracy=sum(row.title_match for row in rows) / count,
        class_top_3_accuracy=sum(
            ranked_hit(row.ranked_classes, row.expected_class, 3) for row in rows
        )
        / count,
        aspect_top_3_accuracy=sum(
            ranked_hit(row.ranked_aspects, row.expected_aspect, 3) for row in rows
        )
        / count,
        title_top_3_accuracy=sum(
            ranked_hit(row.ranked_classes, row.expected_class, 3)
            and ranked_hit(row.ranked_aspects, row.expected_aspect, 3)
            for row in rows
        )
        / count,
        class_abstention_rate=sum(row.class_abstained for row in rows) / count,
        aspect_abstention_rate=sum(row.aspect_abstained for row in rows) / count,
        prediction_error_rate=prediction_error_count / count,
    )


def build_theory_induction_report(
    *,
    variant: str,
    fingerprint: str,
    folds: int,
    rows: Sequence[TheoryScoredPrediction],
) -> TheoryInductionReport:
    count = len(rows) or 1
    fold_summaries = [
        summarize_rows([row for row in rows if row.fold == fold], fold=fold)
        for fold in range(folds)
    ]
    prediction_error_count = sum(row.prediction_error is not None for row in rows)
    return TheoryInductionReport(
        variant=variant,
        dataset_fingerprint=fingerprint,
        folds=folds,
        row_count=len(rows),
        class_accuracy=sum(row.class_match for row in rows) / count,
        aspect_accuracy=sum(row.aspect_match for row in rows) / count,
        title_accuracy=sum(row.title_match for row in rows) / count,
        class_top_3_accuracy=sum(
            ranked_hit(row.ranked_classes, row.expected_class, 3) for row in rows
        )
        / count,
        aspect_top_3_accuracy=sum(
            ranked_hit(row.ranked_aspects, row.expected_aspect, 3) for row in rows
        )
        / count,
        title_top_3_accuracy=sum(
            ranked_hit(row.ranked_classes, row.expected_class, 3)
            and ranked_hit(row.ranked_aspects, row.expected_aspect, 3)
            for row in rows
        )
        / count,
        class_abstention_rate=sum(row.class_abstained for row in rows) / count,
        aspect_abstention_rate=sum(row.aspect_abstained for row in rows) / count,
        prediction_error_count=prediction_error_count,
        prediction_error_rate=prediction_error_count / count,
        fold_summaries=fold_summaries,
        rows=list(rows),
    )


def top_theory_confusions(
    rows: Sequence[TheoryScoredPrediction], target: TheoryTarget, *, limit: int = 10
) -> list[TheoryLabelConfusion]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if target == "class":
            expected = row.expected_class
            predicted = row.predicted_class
        else:
            expected = row.expected_aspect
            predicted = row.predicted_aspect
        if predicted is None or predicted == expected:
            continue
        grouped.setdefault((expected, predicted), []).append(row.character)
    return [
        TheoryLabelConfusion(
            expected=expected,
            predicted=predicted,
            count=len(characters),
            characters=sorted(set(characters)),
        )
        for (expected, predicted), characters in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])
        )[:limit]
    ]


def theory_prediction_bias(
    rows: Sequence[TheoryScoredPrediction], target: TheoryTarget
) -> list[TheoryPredictionBias]:
    labels = labels_for_theory_target(target)
    if target == "class":
        expected_counts = Counter(row.expected_class for row in rows)
        predicted_counts = Counter(row.predicted_class for row in rows if row.predicted_class)
    else:
        expected_counts = Counter(row.expected_aspect for row in rows)
        predicted_counts = Counter(row.predicted_aspect for row in rows if row.predicted_aspect)
    return [
        TheoryPredictionBias(
            label=label,
            expected_count=expected_counts[label],
            predicted_count=predicted_counts[label],
            delta=predicted_counts[label] - expected_counts[label],
        )
        for label in labels
    ]


def build_theory_induction_diagnostics(
    rows: Sequence[TheoryScoredPrediction], *, limit: int = 10
) -> TheoryInductionDiagnostics:
    return TheoryInductionDiagnostics(
        row_count=len(rows),
        top_class_confusions=top_theory_confusions(rows, "class", limit=limit),
        top_aspect_confusions=top_theory_confusions(rows, "aspect", limit=limit),
        class_prediction_bias=theory_prediction_bias(rows, "class"),
        aspect_prediction_bias=theory_prediction_bias(rows, "aspect"),
    )


def load_fold_theory_set(path: Path) -> FoldTheorySet:
    return FoldTheorySet.model_validate_json(path.read_text())


def write_fold_theory_set(path: Path, theory_set: FoldTheorySet) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(theory_set.model_dump_json(indent=2) + "\n")


def theory_source_character_names(theory_set: FoldTheorySet) -> set[str]:
    return {TRAINING_EXAMPLES[character_id].character for character_id in theory_set.train_character_ids}

import json
import os
import re
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel

from experiments.classpect.decision_tree import (
    DESCRIPTIONS_PATH,
    GENERATED_DIR,
    DescriptionRecord,
)
from experiments.classpect.training_examples import TRAINING_EXAMPLES

DEFAULT_DSPY_MODEL = "openai/Qwen3.6-35B-A3B-MTP-GGUF"
DEFAULT_DSPY_API_BASE = os.environ.get("LOCAL_LLM_API_BASE", "http://localhost:8000/v1")
DEFAULT_DSPY_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "")
DSPY_PROGRAM_PATH = GENERATED_DIR / "dspy_classpect_program.json"
DSPY_CLASS_PROGRAM_PATH = GENERATED_DIR / "dspy_class_program.json"
DSPY_ASPECT_PROGRAM_PATH = GENERATED_DIR / "dspy_aspect_program.json"
DSPY_REPORT_PATH = GENERATED_DIR / "dspy_classpect_report.json"
DSPY_FINAL_CLASS_PROGRAM_PATH = GENERATED_DIR / "dspy_final_class_program.json"
DSPY_FINAL_ASPECT_PROGRAM_PATH = GENERATED_DIR / "dspy_final_aspect_program.json"
DSPY_FINAL_REPORT_PATH = GENERATED_DIR / "dspy_final_classpect_report.json"


def dspy_run_suffix(run_name: str | None = None) -> str:
    if not run_name:
        return ""
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_name.strip()).strip("_")
    return f"_{safe_name}" if safe_name else ""


TitleClass = Literal[
    "Bard",
    "Heir",
    "Knight",
    "Mage",
    "Maid",
    "Page",
    "Prince",
    "Rogue",
    "Seer",
    "Sylph",
    "Thief",
    "Witch",
]
TitleAspect = Literal[
    "Blood",
    "Breath",
    "Doom",
    "Heart",
    "Hope",
    "Life",
    "Light",
    "Mind",
    "Rage",
    "Space",
    "Time",
    "Void",
]

VALID_CLASSES: tuple[str, ...] = tuple(
    sorted({example.title_class for example in TRAINING_EXAMPLES})
)
VALID_ASPECTS: tuple[str, ...] = tuple(
    sorted({example.title_aspect for example in TRAINING_EXAMPLES})
)


class DspyEvaluationRow(BaseModel):
    character: str
    character_id: int | None = None
    variant: int | None = None
    expected_class: str
    expected_aspect: str
    predicted_class: str
    predicted_aspect: str
    class_match: bool
    aspect_match: bool
    title_match: bool
    prediction_error: str | None = None


class DspyEvaluationReport(BaseModel):
    fold: int | None = None
    dataset_fingerprint: str | None = None
    folds_total: int | None = None
    train_evaluated: bool = True
    train_size: int
    dev_size: int
    train_class_accuracy: float
    train_aspect_accuracy: float
    train_title_accuracy: float
    dev_class_accuracy: float
    dev_aspect_accuracy: float
    dev_title_accuracy: float
    train_prediction_error_count: int = 0
    train_prediction_error_rate: float = 0.0
    dev_prediction_error_count: int = 0
    dev_prediction_error_rate: float = 0.0
    dev_vote_class_accuracy: float = 0.0
    dev_vote_aspect_accuracy: float = 0.0
    dev_vote_title_accuracy: float = 0.0
    train_rows: list[DspyEvaluationRow]
    dev_rows: list[DspyEvaluationRow]
    dev_vote_rows: list[DspyEvaluationRow] = []
    dev_vote_ties: list[str] = []
    dev_class_confusion: dict[str, dict[str, int]] = {}
    dev_aspect_confusion: dict[str, dict[str, int]] = {}
    dev_vote_class_confusion: dict[str, dict[str, int]] = {}
    dev_vote_aspect_confusion: dict[str, dict[str, int]] = {}
    dev_class_prediction_counts: dict[str, int] = {}
    dev_aspect_prediction_counts: dict[str, int] = {}
    dev_vote_class_prediction_counts: dict[str, int] = {}
    dev_vote_aspect_prediction_counts: dict[str, int] = {}
    dev_class_recall: dict[str, float] = {}
    dev_aspect_recall: dict[str, float] = {}
    dev_vote_class_recall: dict[str, float] = {}
    dev_vote_aspect_recall: dict[str, float] = {}


class DspyCrossValidationReport(BaseModel):
    folds: int
    mean_train_class_accuracy: float
    mean_train_aspect_accuracy: float
    mean_train_title_accuracy: float
    mean_dev_class_accuracy: float
    mean_dev_aspect_accuracy: float
    mean_dev_title_accuracy: float
    mean_train_prediction_error_rate: float = 0.0
    mean_dev_prediction_error_rate: float = 0.0
    mean_dev_vote_class_accuracy: float
    mean_dev_vote_aspect_accuracy: float
    mean_dev_vote_title_accuracy: float
    fold_reports: list[DspyEvaluationReport]


class DspyLabelConfusion(BaseModel):
    expected: str
    predicted: str
    count: int
    characters: list[str]


class DspyPredictionBias(BaseModel):
    label: str
    expected_count: int
    predicted_count: int
    delta: int


class DspyDiagnosticsReport(BaseModel):
    row_count: int
    vote_row_count: int
    top_class_confusions: list[DspyLabelConfusion]
    top_aspect_confusions: list[DspyLabelConfusion]
    top_vote_class_confusions: list[DspyLabelConfusion]
    top_vote_aspect_confusions: list[DspyLabelConfusion]
    class_prediction_bias: list[DspyPredictionBias]
    aspect_prediction_bias: list[DspyPredictionBias]
    vote_class_prediction_bias: list[DspyPredictionBias]
    vote_aspect_prediction_bias: list[DspyPredictionBias]


class DescriptionDatasetSummary(BaseModel):
    path: str
    generated_records: int
    valid_records: int
    invalid_records: int
    characters_represented: int
    missing_characters: list[str]
    minimum_valid_variants_per_character: int
    maximum_valid_variants_per_character: int
    dataset_fingerprint: str | None


class RankedClasspectRow(BaseModel):
    character: str
    character_id: int | None = None
    variant: int | None = None
    expected_class: str
    expected_aspect: str
    ranked_classes: list[str]
    ranked_aspects: list[str]
    abstained: bool = False
    raw_prediction: str = ""
    prediction_error: str | None = None


class RankedClasspectReport(BaseModel):
    dataset_fingerprint: str
    row_count: int
    prediction_error_count: int
    prediction_error_rate: float
    abstention_count: int
    abstention_rate: float
    class_top_1_accuracy: float
    class_top_2_accuracy: float
    class_top_3_accuracy: float
    aspect_top_1_accuracy: float
    aspect_top_2_accuracy: float
    aspect_top_3_accuracy: float
    title_top_1_accuracy: float
    title_top_2_accuracy: float
    title_top_3_accuracy: float
    rows: list[RankedClasspectRow]


class DspyFinalTrainingReport(BaseModel):
    dataset_fingerprint: str
    train_evaluated: bool = True
    train_size: int
    optimizer: str
    train_class_accuracy: float
    train_aspect_accuracy: float
    train_title_accuracy: float
    train_prediction_error_count: int = 0
    train_prediction_error_rate: float = 0.0
    train_vote_class_accuracy: float
    train_vote_aspect_accuracy: float
    train_vote_title_accuracy: float
    train_rows: list[DspyEvaluationRow]
    train_vote_rows: list[DspyEvaluationRow]
    train_vote_ties: list[str]


class ClassSignature(dspy.Signature):
    """Choose the Homestuck class from personality evidence only."""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata."
        )
    )
    title_class: TitleClass = dspy.OutputField(
        desc="One class only. Valid classes: " + ", ".join(VALID_CLASSES)
    )


class AspectSignature(dspy.Signature):
    """Choose the Homestuck aspect from personality evidence only."""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata."
        )
    )
    title_aspect: TitleAspect = dspy.OutputField(
        desc="One aspect only. Valid aspects: " + ", ".join(VALID_ASPECTS)
    )


class NoThinkClassSignature(dspy.Signature):
    """Choose the Homestuck class from personality evidence only. /no_think"""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata. /no_think"
        )
    )
    title_class: TitleClass = dspy.OutputField(
        desc="One class only. Valid classes: " + ", ".join(VALID_CLASSES)
    )


class NoThinkAspectSignature(dspy.Signature):
    """Choose the Homestuck aspect from personality evidence only. /no_think"""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata. /no_think"
        )
    )
    title_aspect: TitleAspect = dspy.OutputField(
        desc="One aspect only. Valid aspects: " + ", ".join(VALID_ASPECTS)
    )


class ConstrainedNoThinkClassSignature(dspy.Signature):
    """Choose the Homestuck class from personality evidence only. /no_think"""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata. /no_think"
        )
    )
    title_class: TitleClass = dspy.OutputField(
        desc=(
            "Return exactly one valid class label and nothing else. Do not explain, "
            "qualify, punctuate, or invent labels. Valid classes: "
            + ", ".join(VALID_CLASSES)
        )
    )


class ConstrainedNoThinkAspectSignature(dspy.Signature):
    """Choose the Homestuck aspect from personality evidence only. /no_think"""

    personality_description: str = dspy.InputField(
        desc=(
            "A personality-only description. Do not infer from blood color, gender, "
            "dream moon, species, weapons, powers, or canonical metadata. /no_think"
        )
    )
    title_aspect: TitleAspect = dspy.OutputField(
        desc=(
            "Return exactly one valid aspect label and nothing else. Do not explain, "
            "qualify, punctuate, or invent labels. Valid aspects: "
            + ", ".join(VALID_ASPECTS)
        )
    )


PredictorStyle = Literal["chain-of-thought", "direct"]


def class_signature(
    no_think: bool = False, *, constrained: bool = False
) -> type[dspy.Signature]:
    if constrained:
        return ConstrainedNoThinkClassSignature
    return NoThinkClassSignature if no_think else ClassSignature


def aspect_signature(
    no_think: bool = False, *, constrained: bool = False
) -> type[dspy.Signature]:
    if constrained:
        return ConstrainedNoThinkAspectSignature
    return NoThinkAspectSignature if no_think else AspectSignature


class ClasspectProgram(dspy.Module):
    def __init__(
        self, *, no_think: bool = False, constrained: bool = False
    ) -> None:
        self.class_predictor = dspy.ChainOfThought(
            class_signature(no_think, constrained=constrained)
        )
        self.aspect_predictor = dspy.ChainOfThought(
            aspect_signature(no_think, constrained=constrained)
        )

    def forward(self, personality_description: str) -> dspy.Prediction:
        class_prediction = self.class_predictor(
            personality_description=personality_description
        )
        aspect_prediction = self.aspect_predictor(
            personality_description=personality_description
        )
        return dspy.Prediction(
            title_class=class_prediction.title_class,
            title_aspect=aspect_prediction.title_aspect,
            class_reasoning=getattr(class_prediction, "reasoning", ""),
            aspect_reasoning=getattr(aspect_prediction, "reasoning", ""),
        )


class ClassProgram(dspy.Module):
    def __init__(
        self, *, no_think: bool = False, constrained: bool = False
    ) -> None:
        self.predict = dspy.ChainOfThought(
            class_signature(no_think, constrained=constrained)
        )

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(personality_description=personality_description)


class AspectProgram(dspy.Module):
    def __init__(
        self, *, no_think: bool = False, constrained: bool = False
    ) -> None:
        self.predict = dspy.ChainOfThought(
            aspect_signature(no_think, constrained=constrained)
        )

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(personality_description=personality_description)


class DirectClassProgram(dspy.Module):
    def __init__(
        self, *, no_think: bool = False, constrained: bool = False
    ) -> None:
        self.predict = dspy.Predict(
            class_signature(no_think, constrained=constrained)
        )

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(personality_description=personality_description)


class DirectAspectProgram(dspy.Module):
    def __init__(
        self, *, no_think: bool = False, constrained: bool = False
    ) -> None:
        self.predict = dspy.Predict(
            aspect_signature(no_think, constrained=constrained)
        )

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(personality_description=personality_description)


def make_class_program(
    predictor: PredictorStyle = "chain-of-thought",
    *,
    no_think: bool = False,
    constrained: bool = False,
) -> dspy.Module:
    if predictor == "chain-of-thought":
        return ClassProgram(no_think=no_think, constrained=constrained)
    if predictor == "direct":
        return DirectClassProgram(no_think=no_think, constrained=constrained)
    raise ValueError(f"Unknown predictor style: {predictor}")


def make_aspect_program(
    predictor: PredictorStyle = "chain-of-thought",
    *,
    no_think: bool = False,
    constrained: bool = False,
) -> dspy.Module:
    if predictor == "chain-of-thought":
        return AspectProgram(no_think=no_think, constrained=constrained)
    if predictor == "direct":
        return DirectAspectProgram(no_think=no_think, constrained=constrained)
    raise ValueError(f"Unknown predictor style: {predictor}")


class SeparateClasspectProgram(dspy.Module):
    def __init__(self, class_program: dspy.Module, aspect_program: dspy.Module) -> None:
        self.class_program = class_program
        self.aspect_program = aspect_program

    def forward(self, personality_description: str) -> dspy.Prediction:
        class_prediction = self.class_program(
            personality_description=personality_description
        )
        aspect_prediction = self.aspect_program(
            personality_description=personality_description
        )
        return dspy.Prediction(
            title_class=class_prediction.title_class,
            title_aspect=aspect_prediction.title_aspect,
            class_reasoning=getattr(class_prediction, "reasoning", ""),
            aspect_reasoning=getattr(aspect_prediction, "reasoning", ""),
        )


def configure_dspy(
    model: str = DEFAULT_DSPY_MODEL,
    *,
    api_base: str = DEFAULT_DSPY_API_BASE,
    api_key: str = DEFAULT_DSPY_API_KEY,
    temperature: float = 0.0,
    max_tokens: int = 16384,
) -> None:
    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": False},
    )
    dspy.configure(lm=lm)


def load_description_records(
    path: Path = DESCRIPTIONS_PATH, *, include_invalid: bool = False
) -> list[DescriptionRecord]:
    records = [
        DescriptionRecord.model_validate(record)
        for record in json.loads(path.read_text())
    ]
    if include_invalid:
        return records
    return [record for record in records if not record.validation_errors]


def build_dspy_examples(
    descriptions: Sequence[DescriptionRecord],
) -> list[dspy.Example]:
    descriptions_by_character: dict[str, list[DescriptionRecord]] = {}
    for record in descriptions:
        descriptions_by_character.setdefault(record.character, []).append(record)
    missing = [
        example.character
        for example in TRAINING_EXAMPLES
        if example.character not in descriptions_by_character
    ]
    if missing:
        raise ValueError(f"Missing descriptions: {missing}")

    return [
        dspy.Example(
            character_id=character_id,
            variant=record.variant,
            personality_description=record.description,
            title_class=example.title_class,
            title_aspect=example.title_aspect,
        ).with_inputs("personality_description")
        for character_id, example in enumerate(TRAINING_EXAMPLES)
        for record in descriptions_by_character[example.character]
    ]


def dataset_fingerprint(examples: Sequence[dspy.Example]) -> str:
    payload = [
        {
            "character_id": getattr(example, "character_id", None),
            "variant": getattr(example, "variant", None),
            "personality_description": example.personality_description,
            "title_class": example.title_class,
            "title_aspect": example.title_aspect,
        }
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize_description_dataset(path: Path = DESCRIPTIONS_PATH) -> DescriptionDatasetSummary:
    all_records = load_description_records(path, include_invalid=True)
    valid_records = [record for record in all_records if not record.validation_errors]
    valid_counts = Counter(record.character for record in valid_records)
    missing_characters = [
        example.character
        for example in TRAINING_EXAMPLES
        if example.character not in valid_counts
    ]
    examples = build_dspy_examples(valid_records) if not missing_characters else []
    return DescriptionDatasetSummary(
        path=str(path),
        generated_records=len(all_records),
        valid_records=len(valid_records),
        invalid_records=len(all_records) - len(valid_records),
        characters_represented=len(valid_counts),
        missing_characters=missing_characters,
        minimum_valid_variants_per_character=min(valid_counts.values())
        if valid_counts
        else 0,
        maximum_valid_variants_per_character=max(valid_counts.values())
        if valid_counts
        else 0,
        dataset_fingerprint=dataset_fingerprint(examples) if examples else None,
    )


def character_name(example: dspy.Example) -> str:
    character_id = getattr(example, "character_id", None)
    if character_id is None:
        return getattr(example, "character", "unknown")
    return TRAINING_EXAMPLES[int(character_id)].character


def strip_example_metadata(
    examples: Sequence[dspy.Example], target: Literal["class", "aspect", "title"]
) -> list[dspy.Example]:
    stripped = []
    for example in examples:
        fields = {"personality_description": example.personality_description}
        if target in {"class", "title"}:
            fields["title_class"] = example.title_class
        if target in {"aspect", "title"}:
            fields["title_aspect"] = example.title_aspect
        stripped.append(dspy.Example(**fields).with_inputs("personality_description"))
    return stripped


def split_examples(
    examples: Sequence[dspy.Example], dev_every: int = 4
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    if dev_every <= 1:
        raise ValueError("dev_every must be greater than 1")
    trainset = [example for i, example in enumerate(examples) if i % dev_every != 0]
    devset = [example for i, example in enumerate(examples) if i % dev_every == 0]
    return trainset, devset


def split_examples_by_fold(
    examples: Sequence[dspy.Example], folds: int, fold: int
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be between 0 and {folds - 1}")
    trainset = [example for i, example in enumerate(examples) if i % folds != fold]
    devset = [example for i, example in enumerate(examples) if i % folds == fold]
    return trainset, devset


def split_examples_grouped_by_fold(
    examples: Sequence[dspy.Example], folds: int, fold: int
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be between 0 and {folds - 1}")

    held_out_characters = {
        i for i, example in enumerate(TRAINING_EXAMPLES) if i % folds == fold
    }
    trainset: list[dspy.Example] = []
    devset: list[dspy.Example] = []
    for example in examples:
        if example.character_id in held_out_characters:
            devset.append(example)
        else:
            trainset.append(example)
    return trainset, devset


def split_examples_grouped_by_observed_character_fold(
    examples: Sequence[dspy.Example], folds: int, fold: int
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be between 0 and {folds - 1}")

    character_ids = sorted(
        {
            int(example.character_id)
            for example in examples
            if getattr(example, "character_id", None) is not None
        }
    )
    held_out_characters = {
        character_id
        for index, character_id in enumerate(character_ids)
        if index % folds == fold
    }
    trainset: list[dspy.Example] = []
    devset: list[dspy.Example] = []
    for example in examples:
        if example.character_id in held_out_characters:
            devset.append(example)
        else:
            trainset.append(example)
    return trainset, devset


def normalize_choice(value: Any, choices: Sequence[str]) -> str:
    text = str(value or "").strip()
    for choice in choices:
        if text.casefold() == choice.casefold():
            return choice

    tokens = set(re.findall(r"[a-z]+", text.casefold()))
    matches = [choice for choice in choices if choice.casefold() in tokens]
    if len(matches) == 1:
        return matches[0]
    return text


def parse_ranked_labels(
    text: str, choices: Sequence[str], *, limit: int = 3
) -> list[str]:
    matches: list[tuple[int, str]] = []
    for choice in choices:
        match = re.search(rf"\b{re.escape(choice)}\b", text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), choice))
    return [choice for _, choice in sorted(matches)[:limit]]


def ranked_hit(row: RankedClasspectRow, target: Literal["class", "aspect"], k: int) -> bool:
    if target == "class":
        return row.expected_class in row.ranked_classes[:k]
    return row.expected_aspect in row.ranked_aspects[:k]


def ranked_accuracy(
    rows: Sequence[RankedClasspectRow], target: Literal["class", "aspect", "title"], k: int
) -> float:
    count = len(rows) or 1
    if target == "class":
        return sum(ranked_hit(row, "class", k) for row in rows) / count
    if target == "aspect":
        return sum(ranked_hit(row, "aspect", k) for row in rows) / count
    return (
        sum(ranked_hit(row, "class", k) and ranked_hit(row, "aspect", k) for row in rows)
        / count
    )


def build_ranked_classpect_report(
    rows: Sequence[RankedClasspectRow], *, fingerprint: str
) -> RankedClasspectReport:
    count = len(rows) or 1
    prediction_error_count_ = sum(row.prediction_error is not None for row in rows)
    abstention_count = sum(row.abstained for row in rows)
    return RankedClasspectReport(
        dataset_fingerprint=fingerprint,
        row_count=len(rows),
        prediction_error_count=prediction_error_count_,
        prediction_error_rate=prediction_error_count_ / count,
        abstention_count=abstention_count,
        abstention_rate=abstention_count / count,
        class_top_1_accuracy=ranked_accuracy(rows, "class", 1),
        class_top_2_accuracy=ranked_accuracy(rows, "class", 2),
        class_top_3_accuracy=ranked_accuracy(rows, "class", 3),
        aspect_top_1_accuracy=ranked_accuracy(rows, "aspect", 1),
        aspect_top_2_accuracy=ranked_accuracy(rows, "aspect", 2),
        aspect_top_3_accuracy=ranked_accuracy(rows, "aspect", 3),
        title_top_1_accuracy=ranked_accuracy(rows, "title", 1),
        title_top_2_accuracy=ranked_accuracy(rows, "title", 2),
        title_top_3_accuracy=ranked_accuracy(rows, "title", 3),
        rows=list(rows),
    )


def score_prediction(example: dspy.Example, prediction: dspy.Prediction) -> float:
    predicted_class = normalize_choice(
        getattr(prediction, "title_class", ""), VALID_CLASSES
    )
    predicted_aspect = normalize_choice(
        getattr(prediction, "title_aspect", ""), VALID_ASPECTS
    )
    class_match = predicted_class == example.title_class
    aspect_match = predicted_aspect == example.title_aspect
    if class_match and aspect_match:
        return 1.0
    if class_match or aspect_match:
        return 0.5
    return 0.0


def title_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    return score_prediction(example, prediction)


def class_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    predicted_class = normalize_choice(
        getattr(prediction, "title_class", ""), VALID_CLASSES
    )
    return float(predicted_class == example.title_class)


def aspect_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    predicted_aspect = normalize_choice(
        getattr(prediction, "title_aspect", ""), VALID_ASPECTS
    )
    return float(predicted_aspect == example.title_aspect)


def evaluate_examples(
    program: dspy.Module,
    examples: Sequence[dspy.Example],
    *,
    label: str | None = None,
    num_threads: int | None = None,
) -> tuple[float, float, float, list[DspyEvaluationRow]]:
    total = len(examples)

    def evaluate_one(i: int, example: dspy.Example) -> DspyEvaluationRow:
        if label:
            print(
                f"evaluating {label} {i}/{total}: {character_name(example)}", flush=True
            )
        try:
            prediction = program(
                personality_description=example.personality_description
            )
        except Exception as error:
            return DspyEvaluationRow(
                character=character_name(example),
                character_id=getattr(example, "character_id", None),
                variant=getattr(example, "variant", None),
                expected_class=example.title_class,
                expected_aspect=example.title_aspect,
                predicted_class="",
                predicted_aspect="",
                class_match=False,
                aspect_match=False,
                title_match=False,
                prediction_error=f"{type(error).__name__}: {error}",
            )
        predicted_class = normalize_choice(
            getattr(prediction, "title_class", ""), VALID_CLASSES
        )
        predicted_aspect = normalize_choice(
            getattr(prediction, "title_aspect", ""), VALID_ASPECTS
        )
        class_match = predicted_class == example.title_class
        aspect_match = predicted_aspect == example.title_aspect
        return DspyEvaluationRow(
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
        )

    if num_threads is not None and num_threads > 1:
        rows_by_index: dict[int, DspyEvaluationRow] = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(evaluate_one, i, example): i
                for i, example in enumerate(examples, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                rows_by_index[index] = future.result()
                if label:
                    print(f"completed {label} {len(rows_by_index)}/{total}", flush=True)
        rows = [rows_by_index[i] for i in range(1, total + 1)]
    else:
        rows = [evaluate_one(i, example) for i, example in enumerate(examples, start=1)]

    count = len(rows) or 1
    return (
        sum(row.class_match for row in rows) / count,
        sum(row.aspect_match for row in rows) / count,
        sum(row.title_match for row in rows) / count,
        rows,
    )


def accuracy_from_rows(rows: Sequence[DspyEvaluationRow]) -> tuple[float, float, float]:
    count = len(rows) or 1
    return (
        sum(row.class_match for row in rows) / count,
        sum(row.aspect_match for row in rows) / count,
        sum(row.title_match for row in rows) / count,
    )


def prediction_error_count(rows: Sequence[DspyEvaluationRow]) -> int:
    return sum(row.prediction_error is not None for row in rows)


def prediction_error_rate(rows: Sequence[DspyEvaluationRow]) -> float:
    if not rows:
        return 0.0
    return prediction_error_count(rows) / len(rows)


def choose_majority(
    rows: Sequence[DspyEvaluationRow],
    attr: Literal["predicted_class", "predicted_aspect"],
) -> tuple[str, bool]:
    counts = Counter(getattr(row, attr) for row in rows)
    if not counts:
        return "", False
    max_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == max_count)
    return winners[0], len(winners) > 1


def vote_rows(
    rows: Sequence[DspyEvaluationRow],
) -> tuple[list[DspyEvaluationRow], list[str]]:
    rows_by_character: dict[str, list[DspyEvaluationRow]] = {}
    for row in rows:
        rows_by_character.setdefault(row.character, []).append(row)

    voted_rows: list[DspyEvaluationRow] = []
    ties: list[str] = []
    for character in sorted(rows_by_character):
        character_rows = rows_by_character[character]
        first = character_rows[0]
        predicted_class, class_tie = choose_majority(character_rows, "predicted_class")
        predicted_aspect, aspect_tie = choose_majority(
            character_rows, "predicted_aspect"
        )
        if class_tie:
            ties.append(f"{character}:class")
        if aspect_tie:
            ties.append(f"{character}:aspect")
        class_match = predicted_class == first.expected_class
        aspect_match = predicted_aspect == first.expected_aspect
        voted_rows.append(
            DspyEvaluationRow(
                character=character,
                character_id=first.character_id,
                variant=None,
                expected_class=first.expected_class,
                expected_aspect=first.expected_aspect,
                predicted_class=predicted_class,
                predicted_aspect=predicted_aspect,
                class_match=class_match,
                aspect_match=aspect_match,
                title_match=class_match and aspect_match,
            )
        )
    return voted_rows, ties


def confusion_matrix(
    rows: Sequence[DspyEvaluationRow], expected_attr: str, predicted_attr: str
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = getattr(row, expected_attr)
        predicted = getattr(row, predicted_attr)
        out.setdefault(expected, {})[predicted] = (
            out.setdefault(expected, {}).get(predicted, 0) + 1
        )
    return out


def prediction_counts(rows: Sequence[DspyEvaluationRow], attr: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(row, attr) for row in rows).items()))


def recall_by_label(
    rows: Sequence[DspyEvaluationRow], expected_attr: str, predicted_attr: str
) -> dict[str, float]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        expected = getattr(row, expected_attr)
        totals[expected] += 1
        if getattr(row, predicted_attr) == expected:
            correct[expected] += 1
    return {
        label: correct[label] / totals[label]
        for label in sorted(totals)
        if totals[label]
    }


def dev_rows_from_report(
    report: DspyEvaluationReport | DspyCrossValidationReport,
    *,
    voted: bool = False,
) -> list[DspyEvaluationRow]:
    if isinstance(report, DspyCrossValidationReport):
        rows: list[DspyEvaluationRow] = []
        for fold_report in report.fold_reports:
            rows.extend(
                fold_report.dev_vote_rows if voted else fold_report.dev_rows
            )
        return rows
    return list(report.dev_vote_rows if voted else report.dev_rows)


def top_label_confusions(
    rows: Sequence[DspyEvaluationRow],
    *,
    expected_attr: Literal["expected_class", "expected_aspect"],
    predicted_attr: Literal["predicted_class", "predicted_aspect"],
    limit: int = 10,
) -> list[DspyLabelConfusion]:
    counts: Counter[tuple[str, str]] = Counter()
    characters_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        expected = getattr(row, expected_attr)
        predicted = getattr(row, predicted_attr)
        if expected == predicted:
            continue
        pair = (expected, predicted)
        counts[pair] += 1
        characters_by_pair.setdefault(pair, set()).add(row.character)

    return [
        DspyLabelConfusion(
            expected=expected,
            predicted=predicted,
            count=count,
            characters=sorted(characters_by_pair[(expected, predicted)]),
        )
        for (expected, predicted), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:limit]
    ]


def prediction_bias(
    rows: Sequence[DspyEvaluationRow],
    labels: Sequence[str],
    *,
    expected_attr: Literal["expected_class", "expected_aspect"],
    predicted_attr: Literal["predicted_class", "predicted_aspect"],
) -> list[DspyPredictionBias]:
    expected_counts = Counter(getattr(row, expected_attr) for row in rows)
    predicted_counts = Counter(getattr(row, predicted_attr) for row in rows)
    return [
        DspyPredictionBias(
            label=label,
            expected_count=expected_counts[label],
            predicted_count=predicted_counts[label],
            delta=predicted_counts[label] - expected_counts[label],
        )
        for label in sorted(labels)
    ]


def build_dspy_diagnostics_report(
    report: DspyEvaluationReport | DspyCrossValidationReport, *, limit: int = 10
) -> DspyDiagnosticsReport:
    rows = dev_rows_from_report(report)
    vote_rows_ = dev_rows_from_report(report, voted=True)
    return DspyDiagnosticsReport(
        row_count=len(rows),
        vote_row_count=len(vote_rows_),
        top_class_confusions=top_label_confusions(
            rows,
            expected_attr="expected_class",
            predicted_attr="predicted_class",
            limit=limit,
        ),
        top_aspect_confusions=top_label_confusions(
            rows,
            expected_attr="expected_aspect",
            predicted_attr="predicted_aspect",
            limit=limit,
        ),
        top_vote_class_confusions=top_label_confusions(
            vote_rows_,
            expected_attr="expected_class",
            predicted_attr="predicted_class",
            limit=limit,
        ),
        top_vote_aspect_confusions=top_label_confusions(
            vote_rows_,
            expected_attr="expected_aspect",
            predicted_attr="predicted_aspect",
            limit=limit,
        ),
        class_prediction_bias=prediction_bias(
            rows,
            VALID_CLASSES,
            expected_attr="expected_class",
            predicted_attr="predicted_class",
        ),
        aspect_prediction_bias=prediction_bias(
            rows,
            VALID_ASPECTS,
            expected_attr="expected_aspect",
            predicted_attr="predicted_aspect",
        ),
        vote_class_prediction_bias=prediction_bias(
            vote_rows_,
            VALID_CLASSES,
            expected_attr="expected_class",
            predicted_attr="predicted_class",
        ),
        vote_aspect_prediction_bias=prediction_bias(
            vote_rows_,
            VALID_ASPECTS,
            expected_attr="expected_aspect",
            predicted_attr="predicted_aspect",
        ),
    )


def evaluate_program(
    program: dspy.Module,
    trainset: Sequence[dspy.Example],
    devset: Sequence[dspy.Example],
    *,
    fold: int | None = None,
    folds_total: int | None = None,
    fingerprint: str | None = None,
    progress: bool = False,
    num_threads: int | None = None,
    evaluate_train: bool = True,
) -> DspyEvaluationReport:
    fold_label = f"fold {fold}" if fold is not None else "run"
    if evaluate_train:
        train_class, train_aspect, train_title, train_rows = evaluate_examples(
            program,
            trainset,
            label=f"{fold_label} train" if progress else None,
            num_threads=num_threads,
        )
    else:
        train_class, train_aspect, train_title, train_rows = 0.0, 0.0, 0.0, []
    dev_class, dev_aspect, dev_title, dev_rows = evaluate_examples(
        program,
        devset,
        label=f"{fold_label} dev" if progress else None,
        num_threads=num_threads,
    )
    dev_vote_rows, dev_vote_ties = vote_rows(dev_rows)
    dev_vote_class, dev_vote_aspect, dev_vote_title = accuracy_from_rows(dev_vote_rows)
    return DspyEvaluationReport(
        fold=fold,
        dataset_fingerprint=fingerprint,
        folds_total=folds_total,
        train_evaluated=evaluate_train,
        train_size=len(trainset),
        dev_size=len(devset),
        train_class_accuracy=train_class,
        train_aspect_accuracy=train_aspect,
        train_title_accuracy=train_title,
        dev_class_accuracy=dev_class,
        dev_aspect_accuracy=dev_aspect,
        dev_title_accuracy=dev_title,
        train_prediction_error_count=prediction_error_count(train_rows),
        train_prediction_error_rate=prediction_error_rate(train_rows),
        dev_prediction_error_count=prediction_error_count(dev_rows),
        dev_prediction_error_rate=prediction_error_rate(dev_rows),
        dev_vote_class_accuracy=dev_vote_class,
        dev_vote_aspect_accuracy=dev_vote_aspect,
        dev_vote_title_accuracy=dev_vote_title,
        train_rows=train_rows,
        dev_rows=dev_rows,
        dev_vote_rows=dev_vote_rows,
        dev_vote_ties=dev_vote_ties,
        dev_class_confusion=confusion_matrix(
            dev_rows, "expected_class", "predicted_class"
        ),
        dev_aspect_confusion=confusion_matrix(
            dev_rows, "expected_aspect", "predicted_aspect"
        ),
        dev_vote_class_confusion=confusion_matrix(
            dev_vote_rows, "expected_class", "predicted_class"
        ),
        dev_vote_aspect_confusion=confusion_matrix(
            dev_vote_rows, "expected_aspect", "predicted_aspect"
        ),
        dev_class_prediction_counts=prediction_counts(dev_rows, "predicted_class"),
        dev_aspect_prediction_counts=prediction_counts(dev_rows, "predicted_aspect"),
        dev_vote_class_prediction_counts=prediction_counts(
            dev_vote_rows, "predicted_class"
        ),
        dev_vote_aspect_prediction_counts=prediction_counts(
            dev_vote_rows, "predicted_aspect"
        ),
        dev_class_recall=recall_by_label(dev_rows, "expected_class", "predicted_class"),
        dev_aspect_recall=recall_by_label(
            dev_rows, "expected_aspect", "predicted_aspect"
        ),
        dev_vote_class_recall=recall_by_label(
            dev_vote_rows, "expected_class", "predicted_class"
        ),
        dev_vote_aspect_recall=recall_by_label(
            dev_vote_rows, "expected_aspect", "predicted_aspect"
        ),
    )


def evaluate_final_training_program(
    program: dspy.Module,
    trainset: Sequence[dspy.Example],
    *,
    fingerprint: str,
    optimizer_name: str,
    progress: bool = False,
    num_threads: int | None = None,
    evaluate_train: bool = True,
) -> DspyFinalTrainingReport:
    if evaluate_train:
        train_class, train_aspect, train_title, train_rows = evaluate_examples(
            program,
            trainset,
            label="final train" if progress else None,
            num_threads=num_threads,
        )
    else:
        train_class, train_aspect, train_title, train_rows = 0.0, 0.0, 0.0, []
    train_vote_rows, train_vote_ties = vote_rows(train_rows)
    train_vote_class, train_vote_aspect, train_vote_title = accuracy_from_rows(
        train_vote_rows
    )
    return DspyFinalTrainingReport(
        dataset_fingerprint=fingerprint,
        train_evaluated=evaluate_train,
        train_size=len(trainset),
        optimizer=optimizer_name,
        train_class_accuracy=train_class,
        train_aspect_accuracy=train_aspect,
        train_title_accuracy=train_title,
        train_prediction_error_count=prediction_error_count(train_rows),
        train_prediction_error_rate=prediction_error_rate(train_rows),
        train_vote_class_accuracy=train_vote_class,
        train_vote_aspect_accuracy=train_vote_aspect,
        train_vote_title_accuracy=train_vote_title,
        train_rows=train_rows,
        train_vote_rows=train_vote_rows,
        train_vote_ties=train_vote_ties,
    )


def aggregate_reports(
    reports: Sequence[DspyEvaluationReport],
) -> DspyCrossValidationReport:
    count = len(reports) or 1
    train_reports = [report for report in reports if report.train_evaluated]
    train_count = len(train_reports) or 1
    return DspyCrossValidationReport(
        folds=len(reports),
        mean_train_class_accuracy=sum(
            report.train_class_accuracy for report in train_reports
        )
        / train_count,
        mean_train_aspect_accuracy=sum(
            report.train_aspect_accuracy for report in train_reports
        )
        / train_count,
        mean_train_title_accuracy=sum(
            report.train_title_accuracy for report in train_reports
        )
        / train_count,
        mean_dev_class_accuracy=sum(report.dev_class_accuracy for report in reports)
        / count,
        mean_dev_aspect_accuracy=sum(report.dev_aspect_accuracy for report in reports)
        / count,
        mean_dev_title_accuracy=sum(report.dev_title_accuracy for report in reports)
        / count,
        mean_train_prediction_error_rate=sum(
            report.train_prediction_error_rate for report in train_reports
        )
        / train_count,
        mean_dev_prediction_error_rate=sum(
            report.dev_prediction_error_rate for report in reports
        )
        / count,
        mean_dev_vote_class_accuracy=sum(
            report.dev_vote_class_accuracy for report in reports
        )
        / count,
        mean_dev_vote_aspect_accuracy=sum(
            report.dev_vote_aspect_accuracy for report in reports
        )
        / count,
        mean_dev_vote_title_accuracy=sum(
            report.dev_vote_title_accuracy for report in reports
        )
        / count,
        fold_reports=list(reports),
    )


def optimize_program(
    trainset: Sequence[dspy.Example],
    *,
    optimizer_name: str = "bootstrap",
    valset: Sequence[dspy.Example] | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
    max_rounds: int = 1,
    num_candidate_programs: int = 8,
    num_trials: int | None = None,
    num_threads: int | None = None,
    no_think: bool = False,
    constrained: bool = False,
) -> dspy.Module:
    if optimizer_name == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=title_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            max_errors=16,
        )
        return optimizer.compile(
            ClasspectProgram(no_think=no_think, constrained=constrained),
            trainset=strip_example_metadata(trainset, "title"),
        )

    if optimizer_name == "random-search":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=title_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            num_candidate_programs=num_candidate_programs,
            num_threads=num_threads,
            max_errors=16,
        )
        return optimizer.compile(
            ClasspectProgram(no_think=no_think, constrained=constrained),
            trainset=strip_example_metadata(trainset, "title"),
            valset=strip_example_metadata(valset or [], "title"),
        )

    if optimizer_name == "mipro":
        validation_examples = list(valset or [])
        optimizer = dspy.MIPROv2(
            metric=title_metric,
            auto=None if num_trials is not None else "light",
            num_candidates=num_candidate_programs if num_trials is not None else None,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            max_errors=16,
        )
        return optimizer.compile(
            ClasspectProgram(no_think=no_think, constrained=constrained),
            trainset=strip_example_metadata(trainset, "title"),
            valset=strip_example_metadata(validation_examples, "title"),
            num_trials=num_trials,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            minibatch_size=min(8, len(validation_examples))
            if validation_examples
            else 1,
        )

    raise ValueError(f"Unknown DSPy optimizer: {optimizer_name}")


def optimize_module(
    module: dspy.Module,
    trainset: Sequence[dspy.Example],
    metric,
    *,
    optimizer_name: str = "mipro",
    valset: Sequence[dspy.Example] | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
    max_rounds: int = 1,
    num_candidate_programs: int = 8,
    num_trials: int | None = 30,
    num_threads: int | None = None,
    target: Literal["class", "aspect", "title"] = "title",
) -> dspy.Module:
    if optimizer_name == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            max_errors=16,
        )
        return optimizer.compile(
            module, trainset=strip_example_metadata(trainset, target)
        )

    if optimizer_name == "random-search":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
            num_candidate_programs=num_candidate_programs,
            num_threads=num_threads,
            max_errors=16,
        )
        return optimizer.compile(
            module,
            trainset=strip_example_metadata(trainset, target),
            valset=strip_example_metadata(valset or [], target),
        )

    if optimizer_name == "mipro":
        validation_examples = list(valset or [])
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None if num_trials is not None else "light",
            num_candidates=num_candidate_programs if num_trials is not None else None,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            max_errors=16,
        )
        return optimizer.compile(
            module,
            trainset=strip_example_metadata(trainset, target),
            valset=strip_example_metadata(validation_examples, target),
            num_trials=num_trials,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            minibatch_size=min(8, len(validation_examples))
            if validation_examples
            else 1,
        )

    raise ValueError(f"Unknown DSPy optimizer: {optimizer_name}")


def optimize_separate_programs(
    trainset: Sequence[dspy.Example],
    *,
    optimizer_name: str = "mipro",
    valset: Sequence[dspy.Example] | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
    max_rounds: int = 1,
    num_candidate_programs: int = 8,
    num_trials: int | None = 30,
    num_threads: int | None = None,
    predictor: PredictorStyle = "chain-of-thought",
    no_think: bool = False,
    constrained: bool = False,
) -> SeparateClasspectProgram:
    class_program = optimize_module(
        make_class_program(predictor, no_think=no_think, constrained=constrained),
        trainset,
        class_metric,
        optimizer_name=optimizer_name,
        valset=valset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        num_candidate_programs=num_candidate_programs,
        num_trials=num_trials,
        num_threads=num_threads,
        target="class",
    )
    aspect_program = optimize_module(
        make_aspect_program(predictor, no_think=no_think, constrained=constrained),
        trainset,
        aspect_metric,
        optimizer_name=optimizer_name,
        valset=valset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        num_candidate_programs=num_candidate_programs,
        num_trials=num_trials,
        num_threads=num_threads,
        target="aspect",
    )
    return SeparateClasspectProgram(class_program, aspect_program)


def dspy_program_path_for_fold(fold: int, run_name: str | None = None) -> Path:
    return (
        GENERATED_DIR
        / f"dspy_classpect_program{dspy_run_suffix(run_name)}_fold_{fold}.json"
    )


def dspy_class_program_path_for_fold(fold: int, run_name: str | None = None) -> Path:
    return (
        GENERATED_DIR
        / f"dspy_class_program{dspy_run_suffix(run_name)}_fold_{fold}.json"
    )


def dspy_aspect_program_path_for_fold(fold: int, run_name: str | None = None) -> Path:
    return (
        GENERATED_DIR
        / f"dspy_aspect_program{dspy_run_suffix(run_name)}_fold_{fold}.json"
    )


def dspy_report_path_for_fold(fold: int, run_name: str | None = None) -> Path:
    return (
        GENERATED_DIR
        / f"dspy_classpect_report{dspy_run_suffix(run_name)}_fold_{fold}.json"
    )


def dspy_metadata_path_for_fold(fold: int, run_name: str | None = None) -> Path:
    return GENERATED_DIR / f"dspy{dspy_run_suffix(run_name)}_fold_{fold}_metadata.json"


def dspy_diagnostics_path(run_name: str | None = None) -> Path:
    return GENERATED_DIR / f"dspy_classpect_diagnostics{dspy_run_suffix(run_name)}.json"


def write_dspy_fold_metadata(
    fold: int,
    *,
    fingerprint: str,
    folds_total: int,
    train_size: int,
    dev_size: int,
    predictor: str = "chain-of-thought",
    no_think: bool = False,
    constrained: bool = False,
    run_name: str | None = None,
) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    dspy_metadata_path_for_fold(fold, run_name).write_text(
        json.dumps(
            {
                "dataset_fingerprint": fingerprint,
                "folds_total": folds_total,
                "train_size": train_size,
                "dev_size": dev_size,
                "predictor": predictor,
                "no_think": no_think,
                "constrained": constrained,
            },
            indent=2,
        )
        + "\n"
    )


def load_dspy_fold_metadata(fold: int, run_name: str | None = None) -> dict[str, Any]:
    return json.loads(dspy_metadata_path_for_fold(fold, run_name).read_text())


def write_dspy_fold_report(
    report: DspyEvaluationReport, run_name: str | None = None
) -> None:
    if report.fold is None:
        raise ValueError("Cannot write fold report without a fold number")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    dspy_report_path_for_fold(report.fold, run_name).write_text(
        report.model_dump_json(indent=2) + "\n"
    )


def load_dspy_fold_report(
    fold: int, run_name: str | None = None
) -> DspyEvaluationReport:
    return DspyEvaluationReport.model_validate_json(
        dspy_report_path_for_fold(fold, run_name).read_text()
    )


def write_dspy_report(report: BaseModel, run_name: str | None = None) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"dspy_classpect_report{dspy_run_suffix(run_name)}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n")


def write_dspy_diagnostics_report(
    report: DspyDiagnosticsReport, run_name: str | None = None
) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    dspy_diagnostics_path(run_name).write_text(report.model_dump_json(indent=2) + "\n")


def write_dspy_final_report(report: DspyFinalTrainingReport) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    DSPY_FINAL_REPORT_PATH.write_text(report.model_dump_json(indent=2) + "\n")

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.classpect.decision_tree import GENERATED_DIR, DescriptionRecord
from experiments.classpect.dspy_classpect import (
    TitleAspect,
    TitleClass,
    VALID_ASPECTS,
    VALID_CLASSES,
    aspect_metric,
    class_metric,
    dataset_fingerprint,
    optimize_module,
)
from experiments.classpect.training_examples import TRAINING_EXAMPLES

LABEL_PROFILES_PATH = GENERATED_DIR / "label_profiles.json"
PROFILE_GUIDED_REPORT_PATH = GENERATED_DIR / "profile_guided_dspy_classpect_report.json"
PROFILE_GUIDED_CLASS_PROGRAM_PATH = GENERATED_DIR / "profile_guided_class_program.json"
PROFILE_GUIDED_ASPECT_PROGRAM_PATH = (
    GENERATED_DIR / "profile_guided_aspect_program.json"
)
PROFILE_SIMILARITY_REPORT_PATH = (
    GENERATED_DIR / "profile_similarity_classpect_report.json"
)

ProfileTarget = Literal["class", "aspect"]


class LabelProfile(BaseModel):
    target: ProfileTarget
    label: str
    source_character_count: int
    source_description_count: int
    mini_descriptions: list[str]


class LabelProfileSet(BaseModel):
    dataset_fingerprint: str | None = None
    fold: int | None = None
    folds_total: int | None = None
    profiles: list[LabelProfile]


class LabelProfileExtractionSignature(dspy.Signature):
    """Extract shared personality evidence from labeled anonymous examples only."""

    label_type: str = dspy.InputField(desc="Either class or aspect.")
    label: str = dspy.InputField(desc="The label shared by the examples.")
    example_descriptions: str = dspy.InputField(
        desc=(
            "Anonymous personality-only descriptions with the same label. Use only "
            "these descriptions, not external Homestuck theory or canon knowledge."
        )
    )
    mini_descriptions: str = dspy.OutputField(
        desc=(
            "Two or three concise bullet mini-descriptions of what these examples "
            "have in common. Avoid names, canon events, species, gender, powers, "
            "weapons, blood color, and plot identifiers."
        )
    )


class ProfileGuidedClassSignature(dspy.Signature):
    """Choose the class using the supplied class mini-descriptions as the rubric."""

    class_profiles: str = dspy.InputField(
        desc="Mini-descriptions for every valid class, derived from anonymous examples."
    )
    personality_description: str = dspy.InputField(
        desc="A personality-only description with no identifying canon metadata."
    )
    title_class: TitleClass = dspy.OutputField(
        desc="One class only. Valid classes: " + ", ".join(VALID_CLASSES)
    )


class ProfileGuidedAspectSignature(dspy.Signature):
    """Choose the aspect using the supplied aspect mini-descriptions as the rubric."""

    aspect_profiles: str = dspy.InputField(
        desc="Mini-descriptions for every valid aspect, derived from anonymous examples."
    )
    personality_description: str = dspy.InputField(
        desc="A personality-only description with no identifying canon metadata."
    )
    title_aspect: TitleAspect = dspy.OutputField(
        desc="One aspect only. Valid aspects: " + ", ".join(VALID_ASPECTS)
    )


class LabelProfileExtractor(dspy.Module):
    def __init__(self) -> None:
        self.predict = dspy.Predict(LabelProfileExtractionSignature)

    def forward(
        self, label_type: str, label: str, example_descriptions: str
    ) -> dspy.Prediction:
        return self.predict(
            label_type=label_type,
            label=label,
            example_descriptions=example_descriptions,
        )


class ProfileGuidedClassProgram(dspy.Module):
    def __init__(self, class_profiles: str) -> None:
        self.class_profiles = class_profiles
        self.predict = dspy.Predict(ProfileGuidedClassSignature)

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(
            class_profiles=self.class_profiles,
            personality_description=personality_description,
        )


class ProfileGuidedAspectProgram(dspy.Module):
    def __init__(self, aspect_profiles: str) -> None:
        self.aspect_profiles = aspect_profiles
        self.predict = dspy.Predict(ProfileGuidedAspectSignature)

    def forward(self, personality_description: str) -> dspy.Prediction:
        return self.predict(
            aspect_profiles=self.aspect_profiles,
            personality_description=personality_description,
        )


class ProfileGuidedClasspectProgram(dspy.Module):
    def __init__(
        self,
        class_program: dspy.Module,
        aspect_program: dspy.Module,
    ) -> None:
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


def label_profiles_path_for_fold(fold: int) -> Path:
    return GENERATED_DIR / f"label_profiles_fold_{fold}.json"


def profile_guided_report_path_for_fold(fold: int) -> Path:
    return GENERATED_DIR / f"profile_guided_dspy_classpect_report_fold_{fold}.json"


def profile_similarity_report_path_for_fold(fold: int) -> Path:
    return GENERATED_DIR / f"profile_similarity_classpect_report_fold_{fold}.json"


def profile_guided_class_program_path_for_fold(fold: int) -> Path:
    return GENERATED_DIR / f"profile_guided_class_program_fold_{fold}.json"


def profile_guided_aspect_program_path_for_fold(fold: int) -> Path:
    return GENERATED_DIR / f"profile_guided_aspect_program_fold_{fold}.json"


def labels_for_profile_target(target: ProfileTarget) -> tuple[str, ...]:
    return VALID_CLASSES if target == "class" else VALID_ASPECTS


def label_for_character(character: str, target: ProfileTarget) -> str:
    examples_by_character = {
        example.character: example for example in TRAINING_EXAMPLES
    }
    example = examples_by_character[character]
    return example.title_class if target == "class" else example.title_aspect


def select_profile_source_descriptions(
    descriptions: Sequence[DescriptionRecord],
    target: ProfileTarget,
    *,
    max_characters_per_label: int = 3,
    variants_per_character: int = 1,
    allowed_character_ids: set[int] | None = None,
) -> dict[str, list[DescriptionRecord]]:
    descriptions_by_character: dict[str, list[DescriptionRecord]] = {}
    for description in descriptions:
        if description.validation_errors:
            continue
        descriptions_by_character.setdefault(description.character, []).append(
            description
        )

    groups = {label: [] for label in labels_for_profile_target(target)}
    character_counts = {label: 0 for label in labels_for_profile_target(target)}
    for character_id, example in enumerate(TRAINING_EXAMPLES):
        if (
            allowed_character_ids is not None
            and character_id not in allowed_character_ids
        ):
            continue
        label = example.title_class if target == "class" else example.title_aspect
        if character_counts[label] >= max_characters_per_label:
            continue
        character_descriptions = sorted(
            descriptions_by_character.get(example.character, []),
            key=lambda record: record.variant,
        )[:variants_per_character]
        if not character_descriptions:
            continue
        groups[label].extend(character_descriptions)
        character_counts[label] += 1
    return groups


def format_profile_examples(records: Sequence[DescriptionRecord]) -> str:
    return "\n\n".join(
        f"Example {index}:\n{record.description}"
        for index, record in enumerate(records, start=1)
    )


def parse_mini_descriptions(value: str, *, limit: int = 3) -> list[str]:
    lines = []
    for raw_line in value.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        if line:
            lines.append(line)
    if not lines and value.strip():
        lines = [value.strip()]
    return lines[:limit]


def format_profile_context(
    profile_set: LabelProfileSet | Sequence[LabelProfile], target: ProfileTarget
) -> str:
    profiles = (
        profile_set.profiles
        if isinstance(profile_set, LabelProfileSet)
        else profile_set
    )
    profiles_by_label: Mapping[str, LabelProfile] = {
        profile.label: profile for profile in profiles if profile.target == target
    }
    sections = []
    for label in labels_for_profile_target(target):
        profile = profiles_by_label.get(label)
        mini_descriptions = profile.mini_descriptions if profile else []
        if not mini_descriptions:
            mini_descriptions = ["No training descriptions available for this label."]
        bullets = "\n".join(f"- {description}" for description in mini_descriptions)
        sections.append(f"{label}:\n{bullets}")
    return "\n\n".join(sections)


def profile_texts_by_label(
    profile_set: LabelProfileSet | Sequence[LabelProfile], target: ProfileTarget
) -> dict[str, str]:
    profiles = (
        profile_set.profiles
        if isinstance(profile_set, LabelProfileSet)
        else profile_set
    )
    profiles_by_label = {
        profile.label: "\n".join(profile.mini_descriptions)
        for profile in profiles
        if profile.target == target
    }
    return {
        label: profiles_by_label.get(label, "No training descriptions available.")
        for label in labels_for_profile_target(target)
    }


def predict_by_profile_similarity(
    personality_description: str,
    profile_set: LabelProfileSet,
    target: ProfileTarget,
) -> str:
    label_texts = profile_texts_by_label(profile_set, target)
    labels = list(label_texts)
    corpus = [personality_description, *[label_texts[label] for label in labels]]
    vectors = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
        corpus
    )
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    best_index = max(
        range(len(labels)), key=lambda index: (scores[index], labels[index])
    )
    return labels[best_index]


def profile_guided_fingerprint(
    examples: Sequence[dspy.Example], profile_set: LabelProfileSet
) -> str:
    payload = {
        "examples": dataset_fingerprint(examples),
        "profiles": profile_set.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_label_profile_set(path: Path) -> LabelProfileSet:
    return LabelProfileSet.model_validate_json(path.read_text())


def write_label_profile_set(path: Path, profile_set: LabelProfileSet) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(profile_set.model_dump_json(indent=2) + "\n")


def optimize_profile_guided_separate_programs(
    trainset: Sequence[dspy.Example],
    profile_set: LabelProfileSet,
    *,
    optimizer_name: str = "mipro",
    valset: Sequence[dspy.Example] | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
    max_rounds: int = 1,
    num_candidate_programs: int = 8,
    num_trials: int | None = 30,
    num_threads: int | None = None,
) -> ProfileGuidedClasspectProgram:
    class_context = format_profile_context(profile_set, "class")
    aspect_context = format_profile_context(profile_set, "aspect")
    class_program = optimize_module(
        ProfileGuidedClassProgram(class_context),
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
        ProfileGuidedAspectProgram(aspect_context),
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
    return ProfileGuidedClasspectProgram(class_program, aspect_program)

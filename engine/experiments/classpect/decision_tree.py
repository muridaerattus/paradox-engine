import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from experiments.classpect.training_examples import TRAINING_EXAMPLES, TrainingExample

GENERATED_DIR = Path(__file__).resolve().parent / "generated"
DESCRIPTIONS_PATH = GENERATED_DIR / "descriptions.json"
FEATURES_PATH = GENERATED_DIR / "features.json"

FEATURE_NAMES: tuple[str, ...] = (
    "personal_agency",
    "social_initiative",
    "emotional_openness",
    "self_presentation",
    "rule_orientation",
    "risk_tolerance",
    "pragmatism",
    "idealism",
    "relational_focus",
    "identity_focus",
    "knowledge_focus",
    "chaos_tolerance",
    "responsibility_drive",
    "secrecy",
    "conflict_directness",
    "change_orientation",
)


class DescriptionRecord(BaseModel):
    character: str
    description: str
    variant: int = 0
    validation_errors: list[str] = []


class FeatureRecord(BaseModel):
    character: str
    features: dict[str, int] = Field(
        description="Feature values on a 1-5 scale, where 1 is low and 5 is high."
    )

    def feature_vector(self) -> list[int]:
        missing = [name for name in FEATURE_NAMES if name not in self.features]
        if missing:
            raise ValueError(f"Missing features for {self.character}: {missing}")

        out_of_range = {
            name: value
            for name, value in self.features.items()
            if name in FEATURE_NAMES and not 1 <= value <= 5
        }
        if out_of_range:
            raise ValueError(
                f"Feature values must be on a 1-5 scale for {self.character}: "
                f"{out_of_range}"
            )

        return [self.features[name] for name in FEATURE_NAMES]


class TreeTrainingResult(BaseModel):
    target: Literal["class", "aspect"]
    training_accuracy: float
    leave_one_out_accuracy: float
    tree_text: str


def ensure_generated_dir() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_generated_dir()
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_feature_records(path: Path = FEATURES_PATH) -> list[FeatureRecord]:
    return [FeatureRecord.model_validate(record) for record in read_json(path)]


def labels_for(target: Literal["class", "aspect"]) -> list[str]:
    if target == "class":
        return [example.title_class for example in TRAINING_EXAMPLES]
    return [example.title_aspect for example in TRAINING_EXAMPLES]


def align_features(
    records: Sequence[FeatureRecord],
    examples: Sequence[TrainingExample] = TRAINING_EXAMPLES,
) -> list[list[int]]:
    records_by_character: Mapping[str, FeatureRecord] = {
        record.character: record for record in records
    }
    missing = [
        example.character
        for example in examples
        if example.character not in records_by_character
    ]
    if missing:
        raise ValueError(f"Missing feature records: {missing}")

    return [
        records_by_character[example.character].feature_vector() for example in examples
    ]


def train_tree(
    records: Sequence[FeatureRecord],
    target: Literal["class", "aspect"],
    *,
    max_depth: int | None = 5,
    random_state: int = 13,
) -> TreeTrainingResult:
    features = align_features(records)
    labels = labels_for(target)
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    classifier.fit(features, labels)

    training_predictions = classifier.predict(features)
    loo_predictions = cross_val_predict(
        DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        features,
        labels,
        cv=LeaveOneOut(),
    )

    return TreeTrainingResult(
        target=target,
        training_accuracy=accuracy_score(labels, training_predictions),
        leave_one_out_accuracy=accuracy_score(labels, loo_predictions),
        tree_text=export_text(classifier, feature_names=list(FEATURE_NAMES)),
    )

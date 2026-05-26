from experiments.classpect.decision_tree import FEATURE_NAMES, FeatureRecord, train_tree
from experiments.classpect.training_examples import TRAINING_EXAMPLES


def test_training_roster_has_expected_examples():
    assert len(TRAINING_EXAMPLES) == 32
    assert sum(1 for example in TRAINING_EXAMPLES if example.group == "kid") == 8
    assert (
        sum(1 for example in TRAINING_EXAMPLES if example.group == "Alternian troll")
        == 12
    )
    assert sum(1 for example in TRAINING_EXAMPLES if example.group == "dancestor") == 12
    assert len({example.character for example in TRAINING_EXAMPLES}) == 32


def test_feature_record_requires_all_features():
    record = FeatureRecord(
        character="John Egbert",
        features={name: 3 for name in FEATURE_NAMES},
    )

    assert record.feature_vector() == [3] * len(FEATURE_NAMES)


def test_train_tree_exports_readable_tree():
    records = []
    for i, example in enumerate(TRAINING_EXAMPLES):
        records.append(
            FeatureRecord(
                character=example.character,
                features={
                    name: ((i + feature_index) % 5) + 1
                    for feature_index, name in enumerate(FEATURE_NAMES)
                },
            )
        )

    result = train_tree(records, "class", max_depth=3)

    assert 0 <= result.training_accuracy <= 1
    assert 0 <= result.leave_one_out_accuracy <= 1
    assert FEATURE_NAMES[0] in result.tree_text

import dspy
import pytest

from experiments.classpect.decision_tree import DescriptionRecord
from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    DEFAULT_DSPY_MODEL,
    DSPY_FINAL_ASPECT_PROGRAM_PATH,
    DSPY_FINAL_CLASS_PROGRAM_PATH,
    ClassSignature,
    ConstrainedNoThinkAspectSignature,
    ConstrainedNoThinkClassSignature,
    DirectClassProgram,
    DspyCrossValidationReport,
    DspyEvaluationReport,
    NoThinkClassSignature,
    aggregate_reports,
    VALID_ASPECTS,
    VALID_CLASSES,
    aspect_signature,
    build_dspy_diagnostics_report,
    build_dspy_examples,
    class_signature,
    dataset_fingerprint,
    dspy_metadata_path_for_fold,
    dspy_run_suffix,
    evaluate_examples,
    evaluate_final_training_program,
    load_description_records,
    load_dspy_fold_metadata,
    make_class_program,
    normalize_choice,
    optimize_program,
    parse_ranked_labels,
    prediction_bias,
    ranked_accuracy,
    RankedClasspectRow,
    score_prediction,
    split_examples,
    split_examples_by_fold,
    split_examples_grouped_by_fold,
    split_examples_grouped_by_observed_character_fold,
    summarize_description_dataset,
    top_label_confusions,
    write_dspy_fold_metadata,
)
from experiments.classpect.scripts.generation.generate_descriptions import (
    description_prompt,
    formatted_messages,
    rejected_terms,
    validate_description,
)
from experiments.classpect.scripts.evaluation.evaluate_pairwise_classpect import (
    PairwiseSpec,
    build_pairwise_report,
    build_pairwise_row,
    matching_examples,
    parse_pair_spec,
    parse_pairwise_choice,
    pairwise_messages,
)
from experiments.classpect.scripts.generation.analyze_invalid_descriptions import (
    analyze_invalid_descriptions,
    invalid_term,
)
import experiments.classpect.dspy_classpect as dspy_classpect
from experiments.classpect.profile_guided import (
    LabelProfile,
    LabelProfileSet,
    format_profile_context,
    parse_mini_descriptions,
    predict_by_profile_similarity,
    profile_guided_fingerprint,
    profile_texts_by_label,
    select_profile_source_descriptions,
)
from experiments.classpect.theory_induction import (
    FoldTheorySet,
    InducedLabelTheory,
    TheoryLabelScore,
    TheoryScoredPrediction,
    build_theory_induction_diagnostics,
    build_theory_induction_report,
    character_ids,
    contrast_examples_for_label,
    extract_json_object,
    fold_theories_path,
    parse_induced_theory,
    parse_theory_scores,
    prediction_margin,
    salvage_json_string_list,
    source_examples_for_label,
    theory_context,
    theory_source_character_names,
)
from experiments.classpect.training_examples import TRAINING_EXAMPLES


def test_dspy_uses_local_openai_compatible_endpoint_by_default():
    import os
    assert DEFAULT_DSPY_MODEL == "openai/Qwen3.6-35B-A3B-MTP-GGUF"
    expected_base = os.environ.get("LOCAL_LLM_API_BASE", "http://localhost:8000/v1")
    assert DEFAULT_DSPY_API_BASE == expected_base
    expected_key = os.environ.get("LOCAL_LLM_API_KEY", "")
    assert DEFAULT_DSPY_API_KEY == expected_key


def test_final_dspy_program_paths_do_not_overwrite_promoted_fold_programs():
    assert DSPY_FINAL_CLASS_PROGRAM_PATH.name == "dspy_final_class_program.json"
    assert DSPY_FINAL_ASPECT_PROGRAM_PATH.name == "dspy_final_aspect_program.json"


def test_class_signature_constrains_output_type():
    assert "Literal" in str(ClassSignature.output_fields["title_class"].annotation)


def test_no_think_signature_includes_explicit_qwen_flag():
    assert "/no_think" in NoThinkClassSignature.__doc__
    assert (
        "/no_think"
        in NoThinkClassSignature.input_fields[
            "personality_description"
        ].json_schema_extra["desc"]
    )


def test_constrained_no_think_signatures_require_exact_labels():
    class_desc = ConstrainedNoThinkClassSignature.output_fields[
        "title_class"
    ].json_schema_extra["desc"]
    aspect_desc = ConstrainedNoThinkAspectSignature.output_fields[
        "title_aspect"
    ].json_schema_extra["desc"]

    assert "/no_think" in ConstrainedNoThinkClassSignature.__doc__
    assert "exactly one valid class label" in class_desc
    assert "nothing else" in class_desc
    assert "Do not explain" in class_desc
    assert "exactly one valid aspect label" in aspect_desc


def test_constrained_signature_helpers_select_constrained_variants():
    assert (
        class_signature(no_think=True, constrained=True)
        is ConstrainedNoThinkClassSignature
    )
    assert (
        aspect_signature(no_think=True, constrained=True)
        is ConstrainedNoThinkAspectSignature
    )


def test_direct_program_factory_uses_direct_predictor():
    program = make_class_program("direct", no_think=True)

    assert isinstance(program, DirectClassProgram)


def test_constrained_direct_program_factory_uses_direct_predictor():
    program = make_class_program("direct", no_think=True, constrained=True)

    assert isinstance(program, DirectClassProgram)


def test_dspy_run_suffix_sanitizes_experiment_names():
    assert dspy_run_suffix("direct no-think") == "_direct_no-think"


def test_build_dspy_examples_from_descriptions():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Personality description for {example.character}.",
        )
        for example in TRAINING_EXAMPLES
    ]

    examples = build_dspy_examples(descriptions)

    assert len(examples) == 32
    assert "character" not in examples[0]
    assert examples[0].character_id == 0
    assert examples[0].title_class == TRAINING_EXAMPLES[0].title_class
    assert examples[0].title_aspect == TRAINING_EXAMPLES[0].title_aspect


def test_build_dspy_examples_rejects_missing_descriptions():
    with pytest.raises(ValueError, match="Missing descriptions"):
        build_dspy_examples([])


def test_load_description_records_accepts_explicit_dataset_path(tmp_path):
    path = tmp_path / "descriptions_v2.json"
    path.write_text(
        '[{"character":"John Egbert","variant":0,"description":"Anonymous description.","validation_errors":[]}]\n'
    )

    records = load_description_records(path)

    assert len(records) == 1
    assert records[0].character == "John Egbert"
    assert records[0].description == "Anonymous description."


def test_contrast_v2_description_prompt_mentions_contrastive_evidence():
    prompt = description_prompt("contrast-v2")

    assert "contrastive evidence" in prompt.messages[0].prompt.template


def test_contrast_v3_description_prompt_mentions_aspect_balance():
    prompt = description_prompt("contrast-v3")

    assert "aspect-like evidence balanced" in prompt.messages[0].prompt.template


def test_personality_plus_abstract_role_prompt_requires_anonymous_role_evidence():
    prompt = description_prompt("personality-plus-abstract-role")
    template = prompt.messages[0].prompt.template

    assert "abstract narrative function" in template
    assert "protagonist" in template
    assert "direct plot identifiers" in template


def test_description_prompt_rejects_unknown_style():
    with pytest.raises(ValueError, match="Unknown description prompt style"):
        description_prompt("not-real")


def test_retry_messages_include_rejected_terms():
    messages = formatted_messages(
        description_prompt("baseline"),
        TRAINING_EXAMPLES[0],
        ["forbidden metadata term: human", "forbidden metadata term: weapon"],
    )

    assert rejected_terms(["forbidden metadata term: human"]) == ["human"]
    assert messages[-1]["role"] == "user"
    assert "previous draft was rejected" in messages[-1]["content"]
    assert "human" in messages[-1]["content"]
    assert "weapon" in messages[-1]["content"]


def test_validate_description_rejects_explicit_story_role_labels():
    errors = validate_description(
        "This person acts like the protagonist and main character.",
        TRAINING_EXAMPLES[0],
    )

    assert "forbidden metadata term: protagonist" in errors
    assert "forbidden metadata term: main character" in errors


def test_dataset_fingerprint_changes_with_description_content():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Personality description for {example.character}.",
        )
        for example in TRAINING_EXAMPLES
    ]
    examples = build_dspy_examples(descriptions)
    changed_examples = build_dspy_examples(
        [
            description.model_copy(
                update={"description": description.description + " Extra detail."}
            )
            if index == 0
            else description
            for index, description in enumerate(descriptions)
        ]
    )

    assert dataset_fingerprint(examples) != dataset_fingerprint(changed_examples)


def test_summarize_description_dataset_reports_counts_and_fingerprint(tmp_path):
    path = tmp_path / "descriptions.json"
    records = [
        {
            "character": example.character,
            "variant": 0,
            "description": f"Description for character {index}.",
            "validation_errors": [],
        }
        for index, example in enumerate(TRAINING_EXAMPLES)
    ]
    records.append(
        {
            "character": TRAINING_EXAMPLES[0].character,
            "variant": 1,
            "description": "Invalid leaked description.",
            "validation_errors": ["forbidden identifying term"],
        }
    )
    path.write_text(dspy_classpect.json.dumps(records) + "\n")

    summary = summarize_description_dataset(path)

    assert summary.generated_records == 33
    assert summary.valid_records == 32
    assert summary.invalid_records == 1
    assert summary.characters_represented == 32
    assert summary.missing_characters == []
    assert summary.minimum_valid_variants_per_character == 1
    assert summary.maximum_valid_variants_per_character == 1
    assert summary.dataset_fingerprint is not None
    assert len(summary.dataset_fingerprint) == 64


def test_invalid_description_analysis_counts_terms_and_characters(tmp_path):
    path = tmp_path / "descriptions.json"
    records = [
        {
            "character": "John Egbert",
            "variant": 0,
            "description": "Valid anonymous description.",
            "validation_errors": [],
        },
        {
            "character": "Rose Lalonde",
            "variant": 0,
            "description": "Invalid anonymous description.",
            "validation_errors": ["forbidden identifying term: human"],
        },
        {
            "character": "Rose Lalonde",
            "variant": 1,
            "description": "Invalid anonymous description.",
            "validation_errors": [
                "forbidden identifying term: weapon",
                "forbidden identifying term: human",
            ],
        },
    ]
    path.write_text(dspy_classpect.json.dumps(records) + "\n")

    report = analyze_invalid_descriptions(path)

    assert invalid_term("forbidden identifying term: human") == "human"
    assert report["generated_records"] == 3
    assert report["invalid_records"] == 2
    assert report["invalid_terms"] == {"human": 2, "weapon": 1}
    assert report["invalid_characters"] == {"Rose Lalonde": 2}
    assert report["terms_by_character"] == {"Rose Lalonde": {"human": 2, "weapon": 1}}


def test_fold_metadata_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(dspy_classpect, "GENERATED_DIR", tmp_path)

    write_dspy_fold_metadata(
        2,
        fingerprint="abc123",
        folds_total=4,
        train_size=24,
        dev_size=8,
        predictor="direct",
        no_think=True,
        constrained=True,
    )

    assert dspy_metadata_path_for_fold(2) == tmp_path / "dspy_fold_2_metadata.json"
    assert load_dspy_fold_metadata(2) == {
        "dataset_fingerprint": "abc123",
        "folds_total": 4,
        "train_size": 24,
        "dev_size": 8,
        "predictor": "direct",
        "no_think": True,
        "constrained": True,
    }


def test_split_examples_uses_held_out_dev_set():
    examples = [dspy.Example(i=i).with_inputs("i") for i in range(32)]

    trainset, devset = split_examples(examples, dev_every=4)

    assert len(trainset) == 24
    assert len(devset) == 8


def test_split_examples_by_fold_rotates_dev_set():
    examples = [dspy.Example(i=i).with_inputs("i") for i in range(32)]

    trainset, devset = split_examples_by_fold(examples, folds=4, fold=2)

    assert len(trainset) == 24
    assert len(devset) == 8
    assert [example.i for example in devset] == [2, 6, 10, 14, 18, 22, 26, 30]


def test_split_examples_grouped_by_fold_prevents_variant_leakage():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Description {variant} for {example.character}.",
            variant=variant,
        )
        for example in TRAINING_EXAMPLES
        for variant in range(3)
    ]
    examples = build_dspy_examples(descriptions)

    trainset, devset = split_examples_grouped_by_fold(examples, folds=4, fold=0)
    train_characters = {example.character_id for example in trainset}
    dev_characters = {example.character_id for example in devset}

    assert len(trainset) == 72
    assert len(devset) == 24
    assert not train_characters & dev_characters


def test_inner_grouped_split_uses_observed_training_characters():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Description {variant} for {example.character}.",
            variant=variant,
        )
        for example in TRAINING_EXAMPLES
        for variant in range(2)
    ]
    examples = build_dspy_examples(descriptions)
    outer_trainset, outer_devset = split_examples_grouped_by_fold(
        examples, folds=4, fold=0
    )

    optimizer_trainset, optimizer_valset = split_examples_grouped_by_observed_character_fold(
        outer_trainset, folds=4, fold=0
    )
    optimizer_train_characters = {example.character_id for example in optimizer_trainset}
    optimizer_val_characters = {example.character_id for example in optimizer_valset}
    outer_dev_characters = {example.character_id for example in outer_devset}

    assert len(outer_trainset) == 48
    assert len(outer_devset) == 16
    assert len(optimizer_trainset) == 36
    assert len(optimizer_valset) == 12
    assert not optimizer_train_characters & optimizer_val_characters
    assert not optimizer_val_characters & outer_dev_characters
    assert not optimizer_train_characters & outer_dev_characters


def test_normalize_choice_accepts_verbose_outputs():
    assert normalize_choice("The class is Knight.", VALID_CLASSES) == "Knight"
    assert normalize_choice("I would choose Light", VALID_ASPECTS) == "Light"


def test_parse_ranked_labels_preserves_mentioned_order():
    labels = parse_ranked_labels("1. Seer\n2. Knight\n3. Page", VALID_CLASSES)

    assert labels == ["Seer", "Knight", "Page"]


def test_ranked_accuracy_scores_top_k_hits():
    rows = [
        RankedClasspectRow(
            character="A",
            expected_class="Knight",
            expected_aspect="Time",
            ranked_classes=["Seer", "Knight", "Page"],
            ranked_aspects=["Light", "Time", "Breath"],
        ),
        RankedClasspectRow(
            character="B",
            expected_class="Mage",
            expected_aspect="Doom",
            ranked_classes=["Seer", "Knight", "Page"],
            ranked_aspects=["Light", "Time", "Breath"],
        ),
    ]

    assert ranked_accuracy(rows, "class", 1) == 0.0
    assert ranked_accuracy(rows, "class", 2) == 0.5
    assert ranked_accuracy(rows, "aspect", 2) == 0.5
    assert ranked_accuracy(rows, "title", 2) == 0.5


def test_score_prediction_gives_partial_credit():
    example = dspy.Example(title_class="Knight", title_aspect="Time")
    prediction = dspy.Prediction(title_class="Knight", title_aspect="Breath")

    assert score_prediction(example, prediction) == 0.5


def test_evaluate_examples_records_prediction_errors():
    class FailingProgram(dspy.Module):
        def forward(self, personality_description: str) -> dspy.Prediction:
            raise ValueError("bad prediction")

    example = dspy.Example(
        character_id=0,
        variant=0,
        personality_description="Description.",
        title_class="Knight",
        title_aspect="Time",
    ).with_inputs("personality_description")

    class_accuracy, aspect_accuracy, title_accuracy, rows = evaluate_examples(
        FailingProgram(), [example]
    )

    assert class_accuracy == 0.0
    assert aspect_accuracy == 0.0
    assert title_accuracy == 0.0
    assert rows[0].prediction_error.startswith("ValueError: bad prediction")


def test_top_label_confusions_counts_repeated_mistakes():
    rows = [
        dspy_classpect.DspyEvaluationRow(
            character="A",
            expected_class="Knight",
            expected_aspect="Time",
            predicted_class="Prince",
            predicted_aspect="Time",
            class_match=False,
            aspect_match=True,
            title_match=False,
        ),
        dspy_classpect.DspyEvaluationRow(
            character="B",
            expected_class="Knight",
            expected_aspect="Time",
            predicted_class="Prince",
            predicted_aspect="Doom",
            class_match=False,
            aspect_match=False,
            title_match=False,
        ),
        dspy_classpect.DspyEvaluationRow(
            character="C",
            expected_class="Seer",
            expected_aspect="Light",
            predicted_class="Seer",
            predicted_aspect="Light",
            class_match=True,
            aspect_match=True,
            title_match=True,
        ),
    ]

    confusions = top_label_confusions(
        rows, expected_attr="expected_class", predicted_attr="predicted_class"
    )

    assert len(confusions) == 1
    assert confusions[0].expected == "Knight"
    assert confusions[0].predicted == "Prince"
    assert confusions[0].count == 2
    assert confusions[0].characters == ["A", "B"]


def test_prediction_bias_reports_over_and_under_prediction():
    rows = [
        dspy_classpect.DspyEvaluationRow(
            character="A",
            expected_class="Knight",
            expected_aspect="Time",
            predicted_class="Prince",
            predicted_aspect="Time",
            class_match=False,
            aspect_match=True,
            title_match=False,
        ),
        dspy_classpect.DspyEvaluationRow(
            character="B",
            expected_class="Knight",
            expected_aspect="Time",
            predicted_class="Prince",
            predicted_aspect="Doom",
            class_match=False,
            aspect_match=False,
            title_match=False,
        ),
    ]

    by_label = {
        item.label: item
        for item in prediction_bias(
            rows,
            ["Knight", "Prince"],
            expected_attr="expected_class",
            predicted_attr="predicted_class",
        )
    }

    assert by_label["Knight"].expected_count == 2
    assert by_label["Knight"].predicted_count == 0
    assert by_label["Knight"].delta == -2
    assert by_label["Prince"].expected_count == 0
    assert by_label["Prince"].predicted_count == 2
    assert by_label["Prince"].delta == 2


def test_build_dspy_diagnostics_report_flattens_fold_reports():
    row = dspy_classpect.DspyEvaluationRow(
        character="A",
        expected_class="Knight",
        expected_aspect="Time",
        predicted_class="Prince",
        predicted_aspect="Doom",
        class_match=False,
        aspect_match=False,
        title_match=False,
    )
    fold_report = DspyEvaluationReport(
        fold=0,
        train_size=1,
        dev_size=1,
        train_class_accuracy=0.0,
        train_aspect_accuracy=0.0,
        train_title_accuracy=0.0,
        dev_class_accuracy=0.0,
        dev_aspect_accuracy=0.0,
        dev_title_accuracy=0.0,
        train_rows=[],
        dev_rows=[row],
        dev_vote_rows=[row],
    )
    report = DspyCrossValidationReport(
        folds=1,
        mean_train_class_accuracy=0.0,
        mean_train_aspect_accuracy=0.0,
        mean_train_title_accuracy=0.0,
        mean_dev_class_accuracy=0.0,
        mean_dev_aspect_accuracy=0.0,
        mean_dev_title_accuracy=0.0,
        mean_dev_vote_class_accuracy=0.0,
        mean_dev_vote_aspect_accuracy=0.0,
        mean_dev_vote_title_accuracy=0.0,
        fold_reports=[fold_report],
    )

    diagnostics = build_dspy_diagnostics_report(report)

    assert diagnostics.row_count == 1
    assert diagnostics.vote_row_count == 1
    assert diagnostics.top_class_confusions[0].expected == "Knight"
    assert diagnostics.top_aspect_confusions[0].predicted == "Doom"


def test_optimize_program_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Unknown DSPy optimizer"):
        optimize_program([], optimizer_name="not-real")


def test_aggregate_reports_averages_fold_metrics():
    reports = [
        DspyEvaluationReport(
            fold=0,
            train_size=24,
            dev_size=8,
            train_class_accuracy=0.5,
            train_aspect_accuracy=0.25,
            train_title_accuracy=0.125,
            dev_class_accuracy=0.25,
            dev_aspect_accuracy=0.5,
            dev_title_accuracy=0.0,
            dev_vote_class_accuracy=0.25,
            dev_vote_aspect_accuracy=0.5,
            dev_vote_title_accuracy=0.0,
            train_rows=[],
            dev_rows=[],
        ),
        DspyEvaluationReport(
            fold=1,
            train_size=24,
            dev_size=8,
            train_class_accuracy=1.0,
            train_aspect_accuracy=0.75,
            train_title_accuracy=0.625,
            dev_class_accuracy=0.75,
            dev_aspect_accuracy=1.0,
            dev_title_accuracy=0.5,
            dev_vote_class_accuracy=0.75,
            dev_vote_aspect_accuracy=1.0,
            dev_vote_title_accuracy=0.5,
            train_rows=[],
            dev_rows=[],
        ),
    ]

    report = aggregate_reports(reports)

    assert report.folds == 2
    assert report.mean_train_class_accuracy == 0.75
    assert report.mean_dev_title_accuracy == 0.25
    assert report.mean_dev_vote_title_accuracy == 0.25


def test_aggregate_reports_omits_skipped_train_metrics():
    reports = [
        DspyEvaluationReport(
            fold=0,
            train_size=24,
            dev_size=8,
            train_class_accuracy=0.5,
            train_aspect_accuracy=0.25,
            train_title_accuracy=0.125,
            dev_class_accuracy=0.25,
            dev_aspect_accuracy=0.5,
            dev_title_accuracy=0.0,
            train_rows=[],
            dev_rows=[],
        ),
        DspyEvaluationReport(
            fold=1,
            train_evaluated=False,
            train_size=24,
            dev_size=8,
            train_class_accuracy=0.0,
            train_aspect_accuracy=0.0,
            train_title_accuracy=0.0,
            dev_class_accuracy=0.75,
            dev_aspect_accuracy=1.0,
            dev_title_accuracy=0.5,
            train_rows=[],
            dev_rows=[],
        ),
    ]

    report = aggregate_reports(reports)

    assert report.folds == 2
    assert report.mean_train_class_accuracy == 0.5
    assert report.mean_train_aspect_accuracy == 0.25
    assert report.mean_train_title_accuracy == 0.125
    assert report.mean_dev_class_accuracy == 0.5


def test_evaluate_final_training_program_reports_train_and_vote_metrics():
    class StaticProgram(dspy.Module):
        def forward(self, personality_description: str) -> dspy.Prediction:
            return dspy.Prediction(title_class="Knight", title_aspect="Time")

    examples = [
        dspy.Example(
            character_id=0,
            variant=0,
            personality_description="First variant.",
            title_class="Knight",
            title_aspect="Time",
        ).with_inputs("personality_description"),
        dspy.Example(
            character_id=0,
            variant=1,
            personality_description="Second variant.",
            title_class="Knight",
            title_aspect="Time",
        ).with_inputs("personality_description"),
    ]

    report = evaluate_final_training_program(
        StaticProgram(), examples, fingerprint="abc123", optimizer_name="bootstrap"
    )

    assert report.dataset_fingerprint == "abc123"
    assert report.optimizer == "bootstrap"
    assert report.train_evaluated
    assert report.train_size == 2
    assert report.train_title_accuracy == 1.0
    assert report.train_vote_title_accuracy == 1.0
    assert len(report.train_vote_rows) == 1


def test_evaluate_final_training_program_can_skip_train_eval():
    report = evaluate_final_training_program(
        dspy.Module(),
        [],
        fingerprint="abc123",
        optimizer_name="bootstrap",
        evaluate_train=False,
    )

    assert not report.train_evaluated
    assert report.train_size == 0
    assert report.train_rows == []


def test_select_profile_source_descriptions_limits_distinct_characters():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Description {variant} for {example.character}.",
            variant=variant,
        )
        for example in TRAINING_EXAMPLES
        for variant in range(2)
    ]

    groups = select_profile_source_descriptions(
        descriptions,
        "class",
        max_characters_per_label=2,
        variants_per_character=1,
    )

    assert sorted(groups) == list(VALID_CLASSES)
    assert all(len(records) <= 2 for records in groups.values())
    assert all(
        len({record.character for record in records}) == len(records)
        for records in groups.values()
    )


def test_parse_mini_descriptions_accepts_bullets_and_numbers():
    raw = """
    - Guarded but responsible under pressure.
    2. Handles conflict through indirect control.
    * Protects a private inner life.
    - Extra item ignored.
    """

    assert parse_mini_descriptions(raw) == [
        "Guarded but responsible under pressure.",
        "Handles conflict through indirect control.",
        "Protects a private inner life.",
    ]


def test_format_profile_context_includes_all_labels_with_placeholders():
    profile_set = LabelProfileSet(
        profiles=[
            LabelProfile(
                target="class",
                label="Knight",
                source_character_count=2,
                source_description_count=2,
                mini_descriptions=["Defensive and duty-bound."],
            )
        ]
    )

    context = format_profile_context(profile_set, "class")

    assert "Knight:\n- Defensive and duty-bound." in context
    assert "Bard:" in context
    assert "No training descriptions available" in context


def test_profile_guided_fingerprint_changes_with_profiles():
    examples = [
        dspy.Example(
            personality_description="x", title_class="Knight", title_aspect="Time"
        )
    ]
    profile_set = LabelProfileSet(
        profiles=[
            LabelProfile(
                target="class",
                label="Knight",
                source_character_count=1,
                source_description_count=1,
                mini_descriptions=["First profile."],
            )
        ]
    )
    changed_profile_set = profile_set.model_copy(
        deep=True,
        update={
            "profiles": [
                profile_set.profiles[0].model_copy(
                    update={"mini_descriptions": ["Changed profile."]}
                )
            ]
        },
    )

    assert profile_guided_fingerprint(
        examples, profile_set
    ) != profile_guided_fingerprint(examples, changed_profile_set)


def test_profile_texts_by_label_includes_missing_labels():
    profile_set = LabelProfileSet(
        profiles=[
            LabelProfile(
                target="aspect",
                label="Time",
                source_character_count=1,
                source_description_count=1,
                mini_descriptions=["Time pressure and durable commitment."],
            )
        ]
    )

    texts = profile_texts_by_label(profile_set, "aspect")

    assert texts["Time"] == "Time pressure and durable commitment."
    assert "No training descriptions" in texts["Light"]


def test_predict_by_profile_similarity_chooses_closest_label():
    profile_set = LabelProfileSet(
        profiles=[
            LabelProfile(
                target="class",
                label="Knight",
                source_character_count=1,
                source_description_count=1,
                mini_descriptions=["protective armor loyalty duty defensive persona"],
            ),
            LabelProfile(
                target="class",
                label="Seer",
                source_character_count=1,
                source_description_count=1,
                mini_descriptions=[
                    "analysis knowledge interpretation detached insight"
                ],
            ),
        ]
    )

    assert (
        predict_by_profile_similarity(
            "Uses analysis and interpretation to find insight.", profile_set, "class"
        )
        == "Seer"
    )


def test_parse_pair_spec_validates_target_and_labels():
    spec = parse_pair_spec("aspect:Breath:Hope")

    assert spec.target == "aspect"
    assert spec.label_a == "Breath"
    assert spec.label_b == "Hope"
    with pytest.raises(ValueError, match="Invalid class label"):
        parse_pair_spec("class:Breath:Hope")


def test_pairwise_messages_require_compact_choice_and_no_leak_policy():
    messages = pairwise_messages(
        target="class",
        label_a="Heir",
        label_b="Knight",
        personality_description="Avoidant but loyal under pressure.",
    )

    assert "Return exactly one token" in messages[0]["content"]
    assert "/no_think" in messages[0]["content"]
    assert "names" in messages[0]["content"]
    assert "Candidate A: Heir" in messages[1]["content"]


def test_parse_pairwise_choice_accepts_single_choice_only():
    assert parse_pairwise_choice("A") == "A"
    assert parse_pairwise_choice("I choose B.") == "B"
    assert parse_pairwise_choice("Tie") == "Tie"
    assert parse_pairwise_choice("Abstain") == "Abstain"
    with pytest.raises(ValueError, match="Expected exactly one"):
        parse_pairwise_choice("A or B")


def test_matching_examples_filters_by_pair_labels():
    examples = [
        dspy.Example(title_class="Heir", title_aspect="Breath"),
        dspy.Example(title_class="Knight", title_aspect="Time"),
        dspy.Example(title_class="Seer", title_aspect="Light"),
    ]

    matches = matching_examples(
        examples, PairwiseSpec(target="class", label_a="Heir", label_b="Knight")
    )

    assert [example.title_class for example in matches] == ["Heir", "Knight"]


def test_pairwise_report_counts_true_sink_tie_abstain_and_errors():
    examples = [
        dspy.Example(
            character_id=0,
            variant=0,
            title_class="Heir",
            title_aspect="Breath",
        ),
        dspy.Example(
            character_id=2,
            variant=0,
            title_class="Knight",
            title_aspect="Time",
        ),
        dspy.Example(
            character_id=0,
            variant=1,
            title_class="Heir",
            title_aspect="Breath",
        ),
        dspy.Example(
            character_id=2,
            variant=1,
            title_class="Knight",
            title_aspect="Time",
        ),
        dspy.Example(
            character_id=0,
            variant=2,
            title_class="Heir",
            title_aspect="Breath",
        ),
    ]
    spec = PairwiseSpec(target="class", label_a="Heir", label_b="Knight")
    rows = [
        build_pairwise_row(spec=spec, example=examples[0], raw="A", error=None),
        build_pairwise_row(spec=spec, example=examples[1], raw="A", error=None),
        build_pairwise_row(spec=spec, example=examples[2], raw="Tie", error=None),
        build_pairwise_row(spec=spec, example=examples[3], raw="Abstain", error=None),
        build_pairwise_row(spec=spec, example=examples[4], raw="not valid", error=None),
    ]

    report = build_pairwise_report(rows, [spec], fingerprint="abc")
    summary = report.pair_summaries[0]

    assert summary.row_count == 5
    assert summary.true_label_win_count == 1
    assert summary.sink_label_win_count == 1
    assert summary.tie_count == 1
    assert summary.abstention_count == 1
    assert summary.prediction_error_count == 1


def test_theory_induction_paths_sanitize_variant_names():
    assert (
        fold_theories_path("personality only", 2).name
        == "fold_theories_personality_only_fold_2.json"
    )


def test_extract_json_object_accepts_markdown_wrapped_json():
    data = extract_json_object('```json\n{"scores": []}\n```')

    assert data == {"scores": []}


def test_parse_induced_theory_uses_schema_fields_and_fallbacks():
    theory = parse_induced_theory(
        '{"common_evidence":["initiates movement"],"contrastive_evidence":"less duty-bound","uncertainty_notes":[]}',
        target="aspect",
        label="Breath",
        source_character_count=2,
        source_description_count=3,
    )

    assert theory.target == "aspect"
    assert theory.label == "Breath"
    assert theory.common_evidence == ["initiates movement"]
    assert theory.contrastive_evidence == ["less duty-bound"]
    assert theory.uncertainty_notes == ["No uncertainty notes provided."]


def test_salvage_json_string_list_handles_truncated_arrays():
    values = salvage_json_string_list(
        '{"common_evidence":["first","second","third","fourth"],"contrastive_evidence":["other"',
        "common_evidence",
    )

    assert values == ["first", "second", "third"]


def test_parse_induced_theory_salvages_malformed_json():
    theory = parse_induced_theory(
        '{"common_evidence":["first","second","third"],"contrastive_evidence":["contrast"',
        target="class",
        label="Heir",
        source_character_count=1,
        source_description_count=1,
    )

    assert theory.common_evidence == ["first", "second", "third"]
    assert theory.contrastive_evidence == ["contrast"]


def test_parse_theory_scores_ranks_valid_labels_and_abstain():
    scores, abstained = parse_theory_scores(
        '{"scores":[{"label":"Knight","score":2},{"label":"Heir","score":4.5},{"label":"NotReal","score":5}],"abstain":true}',
        VALID_CLASSES,
    )

    assert abstained is True
    assert [score.label for score in scores] == ["Heir", "Knight"]
    assert scores[0].score == 4.5


def test_prediction_margin_uses_top_two_scores():
    assert prediction_margin(
        [
            TheoryLabelScore(label="Heir", score=4.0),
            TheoryLabelScore(label="Knight", score=2.5),
        ]
    ) == 1.5


def test_theory_source_and_contrast_examples_respect_fold_split():
    descriptions = [
        DescriptionRecord(
            character=example.character,
            description=f"Description for {example.character}.",
        )
        for example in TRAINING_EXAMPLES
    ]
    examples = build_dspy_examples(descriptions)
    trainset, devset = split_examples_grouped_by_fold(examples, 4, 0)

    heir_sources = source_examples_for_label(trainset, "class", "Heir")
    heir_contrasts = contrast_examples_for_label(trainset, "class", "Heir")
    train_ids = set(character_ids(trainset))
    dev_ids = set(character_ids(devset))

    assert train_ids.isdisjoint(dev_ids)
    assert all(example.character_id in train_ids for example in heir_sources)
    assert all(example.title_class == "Heir" for example in heir_sources)
    assert all(example.character_id in train_ids for example in heir_contrasts)
    assert all(example.title_class != "Heir" for example in heir_contrasts)


def test_theory_context_formats_all_sections():
    theory = InducedLabelTheory(
        target="class",
        label="Heir",
        source_character_count=2,
        source_description_count=2,
        common_evidence=["reactive growth"],
        contrastive_evidence=["less performative defense"],
        uncertainty_notes=["sparse support"],
    )

    context = theory_context([theory], "class")

    assert "Heir" in context
    assert "Common evidence" in context
    assert "reactive growth" in context


def test_theory_source_character_names_come_from_train_ids_only():
    theory_set = FoldTheorySet(
        variant="personality_only",
        fold=0,
        folds_total=4,
        dataset_fingerprint="abc",
        train_character_ids=[1, 2],
        dev_character_ids=[0],
        theories=[],
    )

    assert theory_source_character_names(theory_set) == {
        TRAINING_EXAMPLES[1].character,
        TRAINING_EXAMPLES[2].character,
    }


def test_build_theory_induction_report_counts_accuracy_and_top_k():
    rows = [
        TheoryScoredPrediction(
            fold=0,
            character="A",
            expected_class="Heir",
            expected_aspect="Breath",
            predicted_class="Heir",
            predicted_aspect="Hope",
            class_match=True,
            aspect_match=False,
            title_match=False,
            ranked_classes=[TheoryLabelScore(label="Heir", score=5)],
            ranked_aspects=[
                TheoryLabelScore(label="Hope", score=5),
                TheoryLabelScore(label="Breath", score=4),
            ],
        ),
        TheoryScoredPrediction(
            fold=1,
            character="B",
            expected_class="Knight",
            expected_aspect="Time",
            predicted_class="Page",
            predicted_aspect="Time",
            class_match=False,
            aspect_match=True,
            title_match=False,
            ranked_classes=[
                TheoryLabelScore(label="Page", score=5),
                TheoryLabelScore(label="Knight", score=4),
            ],
            ranked_aspects=[TheoryLabelScore(label="Time", score=5)],
            prediction_error="parse failure",
        ),
    ]

    report = build_theory_induction_report(
        variant="personality_only", fingerprint="abc", folds=2, rows=rows
    )

    assert report.row_count == 2
    assert report.class_accuracy == 0.5
    assert report.aspect_accuracy == 0.5
    assert report.class_top_3_accuracy == 1.0
    assert report.aspect_top_3_accuracy == 1.0
    assert report.prediction_error_count == 1


def test_build_theory_induction_diagnostics_reports_confusions_and_bias():
    rows = [
        TheoryScoredPrediction(
            fold=0,
            character="A",
            expected_class="Heir",
            expected_aspect="Breath",
            predicted_class="Knight",
            predicted_aspect="Hope",
        ),
        TheoryScoredPrediction(
            fold=0,
            character="B",
            expected_class="Heir",
            expected_aspect="Breath",
            predicted_class="Knight",
            predicted_aspect="Breath",
        ),
    ]

    diagnostics = build_theory_induction_diagnostics(rows)

    assert diagnostics.top_class_confusions[0].expected == "Heir"
    assert diagnostics.top_class_confusions[0].predicted == "Knight"
    assert diagnostics.top_class_confusions[0].count == 2
    knight_bias = next(
        bias for bias in diagnostics.class_prediction_bias if bias.label == "Knight"
    )
    assert knight_bias.delta == 2

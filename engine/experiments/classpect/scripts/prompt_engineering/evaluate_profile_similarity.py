import argparse

from experiments.classpect.dspy_classpect import (
    DspyEvaluationReport,
    DspyEvaluationRow,
    accuracy_from_rows,
    aggregate_reports,
    build_dspy_examples,
    character_name,
    confusion_matrix,
    load_description_records,
    prediction_counts,
    recall_by_label,
    split_examples_grouped_by_fold,
    vote_rows,
)
from experiments.classpect.profile_guided import (
    LABEL_PROFILES_PATH,
    PROFILE_SIMILARITY_REPORT_PATH,
    label_profiles_path_for_fold,
    load_label_profile_set,
    predict_by_profile_similarity,
    profile_guided_fingerprint,
    profile_similarity_report_path_for_fold,
)


def evaluate_similarity_fold(
    *, folds: int, fold: int, profile_mode: str
) -> DspyEvaluationReport:
    examples = build_dspy_examples(load_description_records())
    trainset, devset = split_examples_grouped_by_fold(examples, folds, fold)
    profile_path = (
        label_profiles_path_for_fold(fold)
        if profile_mode == "fold"
        else LABEL_PROFILES_PATH
    )
    profile_set = load_label_profile_set(profile_path)
    rows = []
    for example in devset:
        predicted_class = predict_by_profile_similarity(
            example.personality_description, profile_set, "class"
        )
        predicted_aspect = predict_by_profile_similarity(
            example.personality_description, profile_set, "aspect"
        )
        class_match = predicted_class == example.title_class
        aspect_match = predicted_aspect == example.title_aspect
        rows.append(
            DspyEvaluationRow(
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
        )
    dev_class, dev_aspect, dev_title = accuracy_from_rows(rows)
    dev_vote_rows, dev_vote_ties = vote_rows(rows)
    dev_vote_class, dev_vote_aspect, dev_vote_title = accuracy_from_rows(dev_vote_rows)
    return DspyEvaluationReport(
        fold=fold,
        dataset_fingerprint=profile_guided_fingerprint(examples, profile_set),
        folds_total=folds,
        train_evaluated=False,
        train_size=len(trainset),
        dev_size=len(devset),
        train_class_accuracy=0.0,
        train_aspect_accuracy=0.0,
        train_title_accuracy=0.0,
        dev_class_accuracy=dev_class,
        dev_aspect_accuracy=dev_aspect,
        dev_title_accuracy=dev_title,
        dev_vote_class_accuracy=dev_vote_class,
        dev_vote_aspect_accuracy=dev_vote_aspect,
        dev_vote_title_accuracy=dev_vote_title,
        train_rows=[],
        dev_rows=rows,
        dev_vote_rows=dev_vote_rows,
        dev_vote_ties=dev_vote_ties,
        dev_class_confusion=confusion_matrix(rows, "expected_class", "predicted_class"),
        dev_aspect_confusion=confusion_matrix(
            rows, "expected_aspect", "predicted_aspect"
        ),
        dev_vote_class_confusion=confusion_matrix(
            dev_vote_rows, "expected_class", "predicted_class"
        ),
        dev_vote_aspect_confusion=confusion_matrix(
            dev_vote_rows, "expected_aspect", "predicted_aspect"
        ),
        dev_class_prediction_counts=prediction_counts(rows, "predicted_class"),
        dev_aspect_prediction_counts=prediction_counts(rows, "predicted_aspect"),
        dev_vote_class_prediction_counts=prediction_counts(
            dev_vote_rows, "predicted_class"
        ),
        dev_vote_aspect_prediction_counts=prediction_counts(
            dev_vote_rows, "predicted_aspect"
        ),
        dev_class_recall=recall_by_label(rows, "expected_class", "predicted_class"),
        dev_aspect_recall=recall_by_label(rows, "expected_aspect", "predicted_aspect"),
        dev_vote_class_recall=recall_by_label(
            dev_vote_rows, "expected_class", "predicted_class"
        ),
        dev_vote_aspect_recall=recall_by_label(
            dev_vote_rows, "expected_aspect", "predicted_aspect"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate label-profile similarity without LLM calls."
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--profile-mode",
        choices=("fold", "global"),
        default="fold",
        help="Use train-only fold profiles or one global profile file.",
    )
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(args.folds))
    reports = []
    for fold in folds:
        report = evaluate_similarity_fold(
            folds=args.folds, fold=fold, profile_mode=args.profile_mode
        )
        reports.append(report)
        profile_similarity_report_path_for_fold(fold).write_text(
            report.model_dump_json(indent=2) + "\n"
        )
        print(f"fold={fold}")
        print(f"dev_class_accuracy={report.dev_class_accuracy:.3f}")
        print(f"dev_aspect_accuracy={report.dev_aspect_accuracy:.3f}")
        print(f"dev_title_accuracy={report.dev_title_accuracy:.3f}")
        print(f"dev_vote_class_accuracy={report.dev_vote_class_accuracy:.3f}")
        print(f"dev_vote_aspect_accuracy={report.dev_vote_aspect_accuracy:.3f}")
        print(f"dev_vote_title_accuracy={report.dev_vote_title_accuracy:.3f}")

    final_report = reports[0] if len(reports) == 1 else aggregate_reports(reports)
    PROFILE_SIMILARITY_REPORT_PATH.write_text(
        final_report.model_dump_json(indent=2) + "\n"
    )
    if len(reports) > 1:
        print(f"folds={final_report.folds}")
        print(f"mean_dev_class_accuracy={final_report.mean_dev_class_accuracy:.3f}")
        print(f"mean_dev_aspect_accuracy={final_report.mean_dev_aspect_accuracy:.3f}")
        print(f"mean_dev_title_accuracy={final_report.mean_dev_title_accuracy:.3f}")
        print(
            "mean_dev_vote_class_accuracy="
            f"{final_report.mean_dev_vote_class_accuracy:.3f}"
        )
        print(
            "mean_dev_vote_aspect_accuracy="
            f"{final_report.mean_dev_vote_aspect_accuracy:.3f}"
        )
        print(
            "mean_dev_vote_title_accuracy="
            f"{final_report.mean_dev_vote_title_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()

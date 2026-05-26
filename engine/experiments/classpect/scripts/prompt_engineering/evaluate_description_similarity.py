import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.classpect.decision_tree import GENERATED_DIR
from experiments.classpect.dspy_classpect import (
    DspyEvaluationReport,
    DspyEvaluationRow,
    accuracy_from_rows,
    aggregate_reports,
    build_dspy_examples,
    character_name,
    confusion_matrix,
    dataset_fingerprint,
    load_description_records,
    prediction_counts,
    recall_by_label,
    split_examples_grouped_by_fold,
    vote_rows,
)

DESCRIPTION_SIMILARITY_REPORT_PATH = (
    GENERATED_DIR / "description_similarity_report.json"
)


def description_similarity_report_path_for_fold(fold: int):
    return GENERATED_DIR / f"description_similarity_report_fold_{fold}.json"


def evaluate_description_similarity_fold(
    *, folds: int, fold: int
) -> DspyEvaluationReport:
    examples = build_dspy_examples(load_description_records())
    trainset, devset = split_examples_grouped_by_fold(examples, folds, fold)
    corpus = [example.personality_description for example in trainset]
    corpus.extend(example.personality_description for example in devset)
    vectors = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=5000
    ).fit_transform(corpus)
    train_vectors = vectors[: len(trainset)]
    dev_vectors = vectors[len(trainset) :]
    rows = []
    for index, example in enumerate(devset):
        scores = cosine_similarity(
            dev_vectors[index : index + 1], train_vectors
        ).ravel()
        best_index = max(
            range(len(trainset)),
            key=lambda train_index: (scores[train_index], -train_index),
        )
        prediction_source = trainset[best_index]
        class_match = prediction_source.title_class == example.title_class
        aspect_match = prediction_source.title_aspect == example.title_aspect
        rows.append(
            DspyEvaluationRow(
                character=character_name(example),
                character_id=getattr(example, "character_id", None),
                variant=getattr(example, "variant", None),
                expected_class=example.title_class,
                expected_aspect=example.title_aspect,
                predicted_class=prediction_source.title_class,
                predicted_aspect=prediction_source.title_aspect,
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
        dataset_fingerprint=dataset_fingerprint(examples),
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
        description="Evaluate nearest-neighbor similarity over anonymous descriptions."
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(args.folds))
    reports = []
    for fold in folds:
        report = evaluate_description_similarity_fold(folds=args.folds, fold=fold)
        reports.append(report)
        description_similarity_report_path_for_fold(fold).write_text(
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
    DESCRIPTION_SIMILARITY_REPORT_PATH.write_text(
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

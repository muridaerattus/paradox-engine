import argparse
from pathlib import Path

from experiments.classpect.dspy_classpect import (
    DspyCrossValidationReport,
    DspyEvaluationReport,
    build_dspy_diagnostics_report,
    dspy_diagnostics_path,
    dspy_report_path_for_fold,
    dspy_run_suffix,
    write_dspy_diagnostics_report,
)
from experiments.classpect.decision_tree import GENERATED_DIR


def default_report_path(run_name: str | None) -> Path:
    return GENERATED_DIR / f"dspy_classpect_report{dspy_run_suffix(run_name)}.json"


def load_report(path: Path) -> DspyCrossValidationReport | DspyEvaluationReport:
    raw = path.read_text()
    if '"fold_reports"' in raw:
        return DspyCrossValidationReport.model_validate_json(raw)
    return DspyEvaluationReport.model_validate_json(raw)


def print_bias(title: str, entries, *, limit: int) -> None:
    over = sorted(entries, key=lambda item: (-item.delta, item.label))[:limit]
    under = sorted(entries, key=lambda item: (item.delta, item.label))[:limit]
    print(title)
    print("  over: " + ", ".join(f"{item.label}={item.delta:+d}" for item in over))
    print("  under: " + ", ".join(f"{item.label}={item.delta:+d}" for item in under))


def print_confusions(title: str, entries) -> None:
    print(title)
    for item in entries:
        characters = "; ".join(item.characters)
        print(f"  {item.expected} -> {item.predicted}: {item.count} ({characters})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contrast diagnostics for a DSPy classpect report."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run suffix used by train_dspy_classpect.py reports.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Explicit report path. Overrides --run-name.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Diagnose one fold report instead of the aggregate run report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of confusion and bias rows to print.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write diagnostics JSON under experiments/classpect/generated/.",
    )
    args = parser.parse_args()

    if args.report_path is not None:
        report_path = args.report_path
    elif args.fold is not None:
        report_path = dspy_report_path_for_fold(args.fold, args.run_name)
    else:
        report_path = default_report_path(args.run_name)

    report = load_report(report_path)
    diagnostics = build_dspy_diagnostics_report(report, limit=args.limit)

    print(f"report_path={report_path}")
    print(f"row_count={diagnostics.row_count}")
    print(f"vote_row_count={diagnostics.vote_row_count}")
    print_confusions("top_class_confusions", diagnostics.top_class_confusions)
    print_confusions("top_aspect_confusions", diagnostics.top_aspect_confusions)
    print_confusions("top_vote_class_confusions", diagnostics.top_vote_class_confusions)
    print_confusions("top_vote_aspect_confusions", diagnostics.top_vote_aspect_confusions)
    print_bias("class_prediction_bias", diagnostics.class_prediction_bias, limit=args.limit)
    print_bias("aspect_prediction_bias", diagnostics.aspect_prediction_bias, limit=args.limit)
    print_bias(
        "vote_class_prediction_bias",
        diagnostics.vote_class_prediction_bias,
        limit=args.limit,
    )
    print_bias(
        "vote_aspect_prediction_bias",
        diagnostics.vote_aspect_prediction_bias,
        limit=args.limit,
    )

    if args.save:
        if args.report_path is not None or args.fold is not None:
            output_path = report_path.with_name(report_path.stem + "_diagnostics.json")
            output_path.write_text(diagnostics.model_dump_json(indent=2) + "\n")
        else:
            write_dspy_diagnostics_report(diagnostics, args.run_name)
            output_path = dspy_diagnostics_path(args.run_name)
        print(f"diagnostics_path={output_path}")


if __name__ == "__main__":
    main()

import argparse
from collections import Counter
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    DEFAULT_DSPY_MODEL,
    DESCRIPTIONS_PATH,
    DSPY_FINAL_ASPECT_PROGRAM_PATH,
    DSPY_FINAL_CLASS_PROGRAM_PATH,
    DSPY_ASPECT_PROGRAM_PATH,
    DSPY_CLASS_PROGRAM_PATH,
    DSPY_PROGRAM_PATH,
    ClasspectProgram,
    SeparateClasspectProgram,
    aggregate_reports,
    build_dspy_examples,
    configure_dspy,
    dataset_fingerprint,
    dspy_aspect_program_path_for_fold,
    dspy_class_program_path_for_fold,
    dspy_metadata_path_for_fold,
    dspy_program_path_for_fold,
    dspy_report_path_for_fold,
    evaluate_final_training_program,
    evaluate_program,
    load_dspy_fold_metadata,
    load_dspy_fold_report,
    load_description_records,
    make_aspect_program,
    make_class_program,
    optimize_program,
    optimize_separate_programs,
    split_examples_by_fold,
    split_examples_grouped_by_fold,
    split_examples_grouped_by_observed_character_fold,
    write_dspy_fold_metadata,
    write_dspy_fold_report,
    write_dspy_final_report,
    write_dspy_report,
)


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description="Optimize and evaluate a DSPy classpecting program."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DSPY_MODEL,
        help="DSPy/LiteLLM model name.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_DSPY_API_BASE,
        help="OpenAI-compatible API base URL for DSPy/LiteLLM.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_DSPY_API_KEY,
        help="API key for the configured DSPy/LiteLLM endpoint.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum output tokens per DSPy/LiteLLM call.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of rotated train/dev folds to run.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one fold index instead of all folds.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Train one separate class/aspect program on all valid descriptions. "
            "This does not produce a held-out estimate; use CV reports for that."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("bootstrap", "random-search", "mipro"),
        default="mipro",
        help="DSPy optimizer to use. MIPROv2 is the modern prompt optimizer default.",
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=4,
        help="Maximum bootstrapped examples DSPy may add.",
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=16,
        help="Maximum labeled examples DSPy may add.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        help="Bootstrap optimization rounds.",
    )
    parser.add_argument(
        "--num-candidate-programs",
        type=int,
        default=8,
        help="Candidate programs for random-search or MIPRO optimization.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=30,
        help="MIPROv2 trial count. Use 0 for DSPy's light-mode setting.",
    )
    parser.add_argument(
        "--mipro-inner-folds",
        type=int,
        default=0,
        help=(
            "When >1, split each outer train fold by character and use the inner "
            "dev split as MIPRO's valset. This keeps the outer dev fold untouched."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Optimizer worker threads.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Evaluate the base DSPy program without BootstrapFewShot.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Optimize one joint class+aspect program instead of separate programs.",
    )
    parser.add_argument(
        "--predictor",
        choices=("chain-of-thought", "direct"),
        default="chain-of-thought",
        help="Separate class/aspect predictor style. Direct avoids reasoning fields.",
    )
    parser.add_argument(
        "--qwen-no-think",
        action="store_true",
        help="Add Qwen's explicit /no_think flag to DSPy signatures.",
    )
    parser.add_argument(
        "--constrained-labels",
        action="store_true",
        help="Use direct no-think signatures that require exactly one label and no explanation.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Suffix for generated reports/programs. Inferred for non-default predictor/no-think runs.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the optimized DSPy program under experiments/classpect/generated/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-fold reports or saved separate programs when present.",
    )
    parser.add_argument(
        "--skip-train-eval",
        action="store_true",
        help="Evaluate only held-out dev examples; train accuracies are omitted from means.",
    )
    parser.add_argument(
        "--descriptions-path",
        type=Path,
        default=DESCRIPTIONS_PATH,
        help="Description dataset JSON to train/evaluate against.",
    )
    args = parser.parse_args()
    if args.constrained_labels:
        if args.predictor != "direct":
            parser.error("--constrained-labels requires --predictor direct")
        if not args.qwen_no_think:
            parser.error("--constrained-labels requires --qwen-no-think")
        if args.joint:
            parser.error("--constrained-labels cannot be combined with --joint")

    configure_dspy(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    print(f"descriptions_path={args.descriptions_path}", flush=True)
    examples = build_dspy_examples(load_description_records(args.descriptions_path))
    fingerprint = dataset_fingerprint(examples)
    num_trials = None if args.num_trials == 0 else args.num_trials
    run_name = args.run_name
    if run_name is None:
        run_parts = []
        if args.predictor != "chain-of-thought":
            run_parts.append(args.predictor.replace("-", "_"))
        if args.qwen_no_think:
            run_parts.append("no_think")
        if args.constrained_labels:
            run_parts.append("constrained")
        run_name = "_".join(run_parts) or None
    if run_name:
        print(f"run_name={run_name}", flush=True)

    if args.final:
        if args.joint:
            raise ValueError("--final only supports separate class/aspect programs")
        if args.fold is not None:
            raise ValueError("--final cannot be combined with --fold")

        print("starting final all-data training", flush=True)
        if (
            args.resume
            and DSPY_FINAL_CLASS_PROGRAM_PATH.exists()
            and DSPY_FINAL_ASPECT_PROGRAM_PATH.exists()
        ):
            print("loading existing final all-data programs", flush=True)
            class_program = make_class_program(
                args.predictor,
                no_think=args.qwen_no_think,
                constrained=args.constrained_labels,
            )
            aspect_program = make_aspect_program(
                args.predictor,
                no_think=args.qwen_no_think,
                constrained=args.constrained_labels,
            )
            class_program.load(DSPY_FINAL_CLASS_PROGRAM_PATH)
            aspect_program.load(DSPY_FINAL_ASPECT_PROGRAM_PATH)
            program = SeparateClasspectProgram(class_program, aspect_program)
        elif args.no_optimize:
            program = SeparateClasspectProgram(
                make_class_program(
                    args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                ),
                make_aspect_program(
                    args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                ),
            )
        else:
            program = optimize_separate_programs(
                examples,
                optimizer_name=args.optimizer,
                valset=examples,
                max_bootstrapped_demos=args.max_bootstrapped_demos,
                max_labeled_demos=args.max_labeled_demos,
                max_rounds=args.max_rounds,
                num_candidate_programs=args.num_candidate_programs,
                num_trials=num_trials,
                num_threads=args.num_threads,
                predictor=args.predictor,
                no_think=args.qwen_no_think,
                constrained=args.constrained_labels,
            )

        if args.save:
            program.class_program.save(DSPY_FINAL_CLASS_PROGRAM_PATH)
            program.aspect_program.save(DSPY_FINAL_ASPECT_PROGRAM_PATH)

        report = evaluate_final_training_program(
            program,
            examples,
            fingerprint=fingerprint,
            optimizer_name="none" if args.no_optimize else args.optimizer,
            progress=True,
            num_threads=args.num_threads,
            evaluate_train=not args.skip_train_eval,
        )
        write_dspy_final_report(report)
        print(f"train_size={report.train_size}", flush=True)
        if report.train_evaluated:
            print(f"train_class_accuracy={report.train_class_accuracy:.3f}", flush=True)
            print(
                f"train_aspect_accuracy={report.train_aspect_accuracy:.3f}", flush=True
            )
            print(f"train_title_accuracy={report.train_title_accuracy:.3f}", flush=True)
            print(
                f"train_vote_class_accuracy={report.train_vote_class_accuracy:.3f}",
                flush=True,
            )
            print(
                f"train_vote_aspect_accuracy={report.train_vote_aspect_accuracy:.3f}",
                flush=True,
            )
            print(
                f"train_vote_title_accuracy={report.train_vote_title_accuracy:.3f}",
                flush=True,
            )
        else:
            print("train evaluation skipped", flush=True)
        return

    folds = [args.fold] if args.fold is not None else list(range(args.folds))
    reports = []

    for fold in folds:
        print(f"starting fold={fold}", flush=True)
        if args.joint:
            trainset, devset = split_examples_by_fold(examples, args.folds, fold)
        else:
            trainset, devset = split_examples_grouped_by_fold(
                examples, args.folds, fold
            )
        optimizer_trainset = trainset
        optimizer_valset = devset
        if args.optimizer == "mipro" and args.mipro_inner_folds > 1:
            inner_fold = fold % args.mipro_inner_folds
            optimizer_trainset, optimizer_valset = (
                split_examples_grouped_by_observed_character_fold(
                    trainset, args.mipro_inner_folds, inner_fold
                )
            )
            print(
                f"mipro inner split fold={inner_fold}/{args.mipro_inner_folds}: "
                f"optimizer_train={len(optimizer_trainset)} "
                f"optimizer_val={len(optimizer_valset)} "
                f"outer_dev={len(devset)}",
                flush=True,
            )
        if args.resume and dspy_report_path_for_fold(fold, run_name).exists():
            existing_report = load_dspy_fold_report(fold, run_name)
            if (
                existing_report.dataset_fingerprint == fingerprint
                and existing_report.folds_total == args.folds
                and existing_report.train_size == len(trainset)
                and existing_report.dev_size == len(devset)
            ):
                print(f"loading existing fold report for fold={fold}", flush=True)
                reports.append(existing_report)
                continue
            print(
                f"ignoring stale fold report for fold={fold}: "
                f"report train/dev={existing_report.train_size}/{existing_report.dev_size}, "
                f"current train/dev={len(trainset)}/{len(devset)}, "
                f"report fingerprint={existing_report.dataset_fingerprint}, "
                f"current fingerprint={fingerprint}",
                flush=True,
            )
        program = None
        if (
            args.resume
            and not args.joint
            and dspy_class_program_path_for_fold(fold, run_name).exists()
            and dspy_aspect_program_path_for_fold(fold, run_name).exists()
        ):
            metadata_matches = False
            if dspy_metadata_path_for_fold(fold, run_name).exists():
                metadata = load_dspy_fold_metadata(fold, run_name)
                metadata_matches = (
                    metadata.get("dataset_fingerprint") == fingerprint
                    and metadata.get("folds_total") == args.folds
                    and metadata.get("train_size") == len(trainset)
                    and metadata.get("dev_size") == len(devset)
                    and metadata.get("predictor", "chain-of-thought")
                    == args.predictor
                    and metadata.get("no_think", False) == args.qwen_no_think
                    and metadata.get("constrained", False) == args.constrained_labels
                )
            if metadata_matches:
                print(
                    f"loading existing optimized programs for fold={fold}", flush=True
                )
                class_program = make_class_program(
                    args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                )
                aspect_program = make_aspect_program(
                    args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                )
                class_program.load(dspy_class_program_path_for_fold(fold, run_name))
                aspect_program.load(dspy_aspect_program_path_for_fold(fold, run_name))
                program = SeparateClasspectProgram(class_program, aspect_program)
            else:
                print(f"ignoring stale optimized programs for fold={fold}", flush=True)

        if program is None and args.no_optimize:
            print(f"using unoptimized program for fold={fold}", flush=True)
            if args.joint:
                program = ClasspectProgram(
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                )
            else:
                program = SeparateClasspectProgram(
                    make_class_program(
                        args.predictor,
                        no_think=args.qwen_no_think,
                        constrained=args.constrained_labels,
                    ),
                    make_aspect_program(
                        args.predictor,
                        no_think=args.qwen_no_think,
                        constrained=args.constrained_labels,
                    ),
                )
        elif program is None:
            print(f"optimizing fold={fold} with {args.optimizer}", flush=True)
            if args.joint:
                program = optimize_program(
                    optimizer_trainset,
                    optimizer_name=args.optimizer,
                    valset=optimizer_valset,
                    max_bootstrapped_demos=args.max_bootstrapped_demos,
                    max_labeled_demos=args.max_labeled_demos,
                    max_rounds=args.max_rounds,
                    num_candidate_programs=args.num_candidate_programs,
                    num_trials=num_trials,
                    num_threads=args.num_threads,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                )
            else:
                program = optimize_separate_programs(
                    optimizer_trainset,
                    optimizer_name=args.optimizer,
                    valset=optimizer_valset,
                    max_bootstrapped_demos=args.max_bootstrapped_demos,
                    max_labeled_demos=args.max_labeled_demos,
                    max_rounds=args.max_rounds,
                    num_candidate_programs=args.num_candidate_programs,
                    num_trials=num_trials,
                    num_threads=args.num_threads,
                    predictor=args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                )

        if args.save:
            if isinstance(program, SeparateClasspectProgram):
                if len(folds) == 1:
                    if run_name is None:
                        program.class_program.save(DSPY_CLASS_PROGRAM_PATH)
                        program.aspect_program.save(DSPY_ASPECT_PROGRAM_PATH)
                program.class_program.save(
                    dspy_class_program_path_for_fold(fold, run_name)
                )
                program.aspect_program.save(
                    dspy_aspect_program_path_for_fold(fold, run_name)
                )
                write_dspy_fold_metadata(
                    fold,
                    fingerprint=fingerprint,
                    folds_total=args.folds,
                    train_size=len(trainset),
                    dev_size=len(devset),
                    predictor=args.predictor,
                    no_think=args.qwen_no_think,
                    constrained=args.constrained_labels,
                    run_name=run_name,
                )
            else:
                if len(folds) == 1:
                    if run_name is None:
                        program.save(DSPY_PROGRAM_PATH)
                program.save(dspy_program_path_for_fold(fold, run_name))

        print(f"evaluating fold={fold}", flush=True)
        report = evaluate_program(
            program,
            trainset,
            devset,
            fold=fold,
            folds_total=args.folds,
            fingerprint=fingerprint,
            progress=True,
            num_threads=args.num_threads,
            evaluate_train=not args.skip_train_eval,
        )
        reports.append(report)
        write_dspy_fold_report(report, run_name)

        print(f"fold={fold}", flush=True)
        print(f"train_size={report.train_size}", flush=True)
        print(f"dev_size={report.dev_size}", flush=True)
        print(f"train_class_accuracy={report.train_class_accuracy:.3f}", flush=True)
        print(f"train_aspect_accuracy={report.train_aspect_accuracy:.3f}", flush=True)
        print(f"train_title_accuracy={report.train_title_accuracy:.3f}", flush=True)
        print(f"dev_class_accuracy={report.dev_class_accuracy:.3f}", flush=True)
        print(f"dev_aspect_accuracy={report.dev_aspect_accuracy:.3f}", flush=True)
        print(f"dev_title_accuracy={report.dev_title_accuracy:.3f}", flush=True)
        print(
            f"dev_prediction_error_rate={report.dev_prediction_error_rate:.3f}",
            flush=True,
        )
        print(
            f"dev_vote_class_accuracy={report.dev_vote_class_accuracy:.3f}",
            flush=True,
        )
        print(
            f"dev_vote_aspect_accuracy={report.dev_vote_aspect_accuracy:.3f}",
            flush=True,
        )
        print(
            f"dev_vote_title_accuracy={report.dev_vote_title_accuracy:.3f}",
            flush=True,
        )
        print(
            f"dev_class_prediction_counts={report.dev_class_prediction_counts}",
            flush=True,
        )
        print(
            f"dev_aspect_prediction_counts={report.dev_aspect_prediction_counts}",
            flush=True,
        )
        print(
            "dev_vote_class_prediction_counts="
            f"{report.dev_vote_class_prediction_counts}",
            flush=True,
        )
        print(
            "dev_vote_aspect_prediction_counts="
            f"{report.dev_vote_aspect_prediction_counts}",
            flush=True,
        )

    if len(reports) == 1:
        final_report = reports[0]
    else:
        final_report = aggregate_reports(reports)
        print(f"folds={final_report.folds}")
        print(f"mean_train_class_accuracy={final_report.mean_train_class_accuracy:.3f}")
        print(
            f"mean_train_aspect_accuracy={final_report.mean_train_aspect_accuracy:.3f}"
        )
        print(f"mean_train_title_accuracy={final_report.mean_train_title_accuracy:.3f}")
        print(f"mean_dev_class_accuracy={final_report.mean_dev_class_accuracy:.3f}")
        print(f"mean_dev_aspect_accuracy={final_report.mean_dev_aspect_accuracy:.3f}")
        print(f"mean_dev_title_accuracy={final_report.mean_dev_title_accuracy:.3f}")
        print(
            "mean_dev_prediction_error_rate="
            f"{final_report.mean_dev_prediction_error_rate:.3f}"
        )
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
        class_counts = Counter()
        aspect_counts = Counter()
        vote_class_counts = Counter()
        vote_aspect_counts = Counter()
        for report in reports:
            class_counts.update(report.dev_class_prediction_counts)
            aspect_counts.update(report.dev_aspect_prediction_counts)
            vote_class_counts.update(report.dev_vote_class_prediction_counts)
            vote_aspect_counts.update(report.dev_vote_aspect_prediction_counts)
        print(f"dev_class_prediction_counts={dict(sorted(class_counts.items()))}")
        print(f"dev_aspect_prediction_counts={dict(sorted(aspect_counts.items()))}")
        print(
            "dev_vote_class_prediction_counts="
            f"{dict(sorted(vote_class_counts.items()))}"
        )
        print(
            "dev_vote_aspect_prediction_counts="
            f"{dict(sorted(vote_aspect_counts.items()))}"
        )

    write_dspy_report(final_report, run_name)


if __name__ == "__main__":
    main()

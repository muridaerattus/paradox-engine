import argparse

from dotenv import find_dotenv, load_dotenv

from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    DEFAULT_DSPY_MODEL,
    aggregate_reports,
    build_dspy_examples,
    configure_dspy,
    evaluate_program,
    load_description_records,
    split_examples_grouped_by_fold,
)
from experiments.classpect.profile_guided import (
    LABEL_PROFILES_PATH,
    PROFILE_GUIDED_ASPECT_PROGRAM_PATH,
    PROFILE_GUIDED_CLASS_PROGRAM_PATH,
    PROFILE_GUIDED_REPORT_PATH,
    ProfileGuidedAspectProgram,
    ProfileGuidedClassProgram,
    ProfileGuidedClasspectProgram,
    format_profile_context,
    label_profiles_path_for_fold,
    load_label_profile_set,
    optimize_profile_guided_separate_programs,
    profile_guided_aspect_program_path_for_fold,
    profile_guided_class_program_path_for_fold,
    profile_guided_fingerprint,
    profile_guided_report_path_for_fold,
)


def write_report(path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description="Optimize and evaluate DSPy classpect programs with label profiles."
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
        help="Number of grouped train/dev folds to run.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one fold index instead of all folds.",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("fold", "global"),
        default="fold",
        help=(
            "Use train-only fold profiles, or reuse the global profile file. "
            "Global mode is exploratory and leaks held-out descriptions."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("bootstrap", "random-search", "mipro"),
        default="mipro",
        help="DSPy optimizer to use.",
    )
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--num-candidate-programs", type=int, default=8)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=30,
        help="MIPROv2 trial count. Use 0 for DSPy's light-mode setting.",
    )
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Evaluate profile-guided base programs without optimization.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save optimized profile-guided class/aspect programs.",
    )
    parser.add_argument(
        "--skip-train-eval",
        action="store_true",
        help="Evaluate only held-out dev examples.",
    )
    args = parser.parse_args()

    configure_dspy(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    examples = build_dspy_examples(load_description_records())
    folds = [args.fold] if args.fold is not None else list(range(args.folds))
    num_trials = None if args.num_trials == 0 else args.num_trials
    reports = []

    for fold in folds:
        print(f"starting fold={fold}", flush=True)
        trainset, devset = split_examples_grouped_by_fold(examples, args.folds, fold)
        profile_path = (
            label_profiles_path_for_fold(fold)
            if args.profile_mode == "fold"
            else LABEL_PROFILES_PATH
        )
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Missing {profile_path}. Generate it with "
                "experiments.classpect.scripts.generate_label_profiles first."
            )
        profile_set = load_label_profile_set(profile_path)
        fingerprint = profile_guided_fingerprint(examples, profile_set)

        if args.no_optimize:
            print(f"using unoptimized profile-guided programs for fold={fold}")
            program = ProfileGuidedClasspectProgram(
                ProfileGuidedClassProgram(format_profile_context(profile_set, "class")),
                ProfileGuidedAspectProgram(
                    format_profile_context(profile_set, "aspect")
                ),
            )
        else:
            print(f"optimizing fold={fold} with {args.optimizer}", flush=True)
            program = optimize_profile_guided_separate_programs(
                trainset,
                profile_set,
                optimizer_name=args.optimizer,
                valset=devset,
                max_bootstrapped_demos=args.max_bootstrapped_demos,
                max_labeled_demos=args.max_labeled_demos,
                max_rounds=args.max_rounds,
                num_candidate_programs=args.num_candidate_programs,
                num_trials=num_trials,
                num_threads=args.num_threads,
            )

        if args.save:
            if len(folds) == 1:
                program.class_program.save(PROFILE_GUIDED_CLASS_PROGRAM_PATH)
                program.aspect_program.save(PROFILE_GUIDED_ASPECT_PROGRAM_PATH)
            program.class_program.save(profile_guided_class_program_path_for_fold(fold))
            program.aspect_program.save(
                profile_guided_aspect_program_path_for_fold(fold)
            )

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
        write_report(profile_guided_report_path_for_fold(fold), report)

        print(f"fold={fold}", flush=True)
        print(f"train_size={report.train_size}", flush=True)
        print(f"dev_size={report.dev_size}", flush=True)
        print(f"dev_class_accuracy={report.dev_class_accuracy:.3f}", flush=True)
        print(f"dev_aspect_accuracy={report.dev_aspect_accuracy:.3f}", flush=True)
        print(f"dev_title_accuracy={report.dev_title_accuracy:.3f}", flush=True)
        print(
            f"dev_vote_class_accuracy={report.dev_vote_class_accuracy:.3f}",
            flush=True,
        )
        print(
            f"dev_vote_aspect_accuracy={report.dev_vote_aspect_accuracy:.3f}",
            flush=True,
        )
        print(
            f"dev_vote_title_accuracy={report.dev_vote_title_accuracy:.3f}", flush=True
        )

    final_report = reports[0] if len(reports) == 1 else aggregate_reports(reports)
    write_report(PROFILE_GUIDED_REPORT_PATH, final_report)
    if len(reports) > 1:
        print(f"folds={final_report.folds}", flush=True)
        print(f"mean_dev_class_accuracy={final_report.mean_dev_class_accuracy:.3f}")
        print(f"mean_dev_aspect_accuracy={final_report.mean_dev_aspect_accuracy:.3f}")
        print(f"mean_dev_title_accuracy={final_report.mean_dev_title_accuracy:.3f}")


if __name__ == "__main__":
    main()

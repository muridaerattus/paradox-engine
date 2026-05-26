import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import find_dotenv, load_dotenv

from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    DEFAULT_DSPY_MODEL,
    build_dspy_examples,
    configure_dspy,
    dataset_fingerprint,
    load_description_records,
)
from experiments.classpect.profile_guided import (
    LABEL_PROFILES_PATH,
    LabelProfile,
    LabelProfileExtractor,
    LabelProfileSet,
    ProfileTarget,
    format_profile_examples,
    label_profiles_path_for_fold,
    labels_for_profile_target,
    parse_mini_descriptions,
    select_profile_source_descriptions,
    write_label_profile_set,
)
from experiments.classpect.training_examples import TRAINING_EXAMPLES


def target_list(target: str) -> list[ProfileTarget]:
    if target == "both":
        return ["class", "aspect"]
    return [target]  # type: ignore[list-item]


def source_character_count(records) -> int:
    return len({record.character for record in records})


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description=(
            "Generate shared class/aspect mini-descriptions from anonymous "
            "personality descriptions."
        )
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
        "--target",
        choices=("class", "aspect", "both"),
        default="both",
        help="Which label profile type to generate.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of grouped folds used when --fold is set.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Generate train-only profiles for one held-out fold.",
    )
    parser.add_argument(
        "--max-characters-per-label",
        type=int,
        default=3,
        help="Maximum distinct characters to summarize per class/aspect.",
    )
    parser.add_argument(
        "--variants-per-character",
        type=int,
        default=1,
        help="Maximum description variants to include per selected character.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of label summaries to request in parallel.",
    )
    args = parser.parse_args()

    descriptions = load_description_records()
    examples = build_dspy_examples(descriptions)
    fingerprint = dataset_fingerprint(examples)
    if args.fold is not None:
        if args.folds <= 1:
            raise ValueError("--folds must be greater than 1")
        if not 0 <= args.fold < args.folds:
            raise ValueError(f"--fold must be between 0 and {args.folds - 1}")
        allowed_character_ids = {
            index
            for index, _ in enumerate(TRAINING_EXAMPLES)
            if index % args.folds != args.fold
        }
        output_path = label_profiles_path_for_fold(args.fold)
    else:
        allowed_character_ids = None
        output_path = LABEL_PROFILES_PATH

    configure_dspy(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    extractor = LabelProfileExtractor()
    jobs = []
    for profile_target in target_list(args.target):
        groups = select_profile_source_descriptions(
            descriptions,
            profile_target,
            max_characters_per_label=args.max_characters_per_label,
            variants_per_character=args.variants_per_character,
            allowed_character_ids=allowed_character_ids,
        )
        for label in labels_for_profile_target(profile_target):
            jobs.append((profile_target, label, groups[label]))

    def generate_one(job) -> LabelProfile:
        profile_target, label, records = job
        if not records:
            return LabelProfile(
                target=profile_target,
                label=label,
                source_character_count=0,
                source_description_count=0,
                mini_descriptions=[
                    "No training descriptions were available for this label in this fold."
                ],
            )
        prediction = extractor(
            label_type=profile_target,
            label=label,
            example_descriptions=format_profile_examples(records),
        )
        mini_descriptions = parse_mini_descriptions(
            getattr(prediction, "mini_descriptions", "")
        )
        return LabelProfile(
            target=profile_target,
            label=label,
            source_character_count=source_character_count(records),
            source_description_count=len(records),
            mini_descriptions=mini_descriptions,
        )

    profiles = []
    if args.num_threads > 1:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {executor.submit(generate_one, job): job for job in jobs}
            for future in as_completed(futures):
                profile_target, label, _ = futures[future]
                profile = future.result()
                print(f"generated {profile_target} profile for {label}", flush=True)
                profiles.append(profile)
    else:
        for job in jobs:
            profile_target, label, _ = job
            profiles.append(generate_one(job))
            print(f"generated {profile_target} profile for {label}", flush=True)

    profile_set = LabelProfileSet(
        dataset_fingerprint=fingerprint,
        fold=args.fold,
        folds_total=args.folds if args.fold is not None else None,
        profiles=sorted(profiles, key=lambda profile: (profile.target, profile.label)),
    )
    write_label_profile_set(output_path, profile_set)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()

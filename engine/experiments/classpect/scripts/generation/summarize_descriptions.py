import argparse
from pathlib import Path

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH
from experiments.classpect.dspy_classpect import summarize_description_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a generated classpect description dataset."
    )
    parser.add_argument(
        "--descriptions-path",
        type=Path,
        default=DESCRIPTIONS_PATH,
        help="Description dataset JSON to summarize.",
    )
    args = parser.parse_args()

    summary = summarize_description_dataset(args.descriptions_path)
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

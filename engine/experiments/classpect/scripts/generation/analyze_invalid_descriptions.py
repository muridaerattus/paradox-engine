import argparse
from collections import Counter, defaultdict
from pathlib import Path

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH
from experiments.classpect.dspy_classpect import load_description_records


def invalid_term(error: str) -> str:
    return error.split(":", 1)[-1].strip() if ":" in error else error.strip()


def analyze_invalid_descriptions(path: Path) -> dict:
    records = load_description_records(path, include_invalid=True)
    invalid_records = [record for record in records if record.validation_errors]
    term_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    terms_by_character: dict[str, Counter[str]] = defaultdict(Counter)
    for record in invalid_records:
        character_counts[record.character] += 1
        for error in record.validation_errors:
            term = invalid_term(error)
            term_counts[term] += 1
            terms_by_character[record.character][term] += 1

    return {
        "path": str(path),
        "generated_records": len(records),
        "invalid_records": len(invalid_records),
        "invalid_rate": len(invalid_records) / (len(records) or 1),
        "invalid_terms": dict(sorted(term_counts.items(), key=lambda item: (-item[1], item[0]))),
        "invalid_characters": dict(
            sorted(character_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "terms_by_character": {
            character: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for character, counts in sorted(terms_by_character.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze invalid generated classpect descriptions."
    )
    parser.add_argument(
        "--descriptions-path",
        type=Path,
        default=DESCRIPTIONS_PATH,
        help="Description dataset JSON to analyze.",
    )
    args = parser.parse_args()

    report = analyze_invalid_descriptions(args.descriptions_path)
    print(f"path={report['path']}")
    print(f"generated_records={report['generated_records']}")
    print(f"invalid_records={report['invalid_records']}")
    print(f"invalid_rate={report['invalid_rate']:.3f}")
    print(f"invalid_terms={report['invalid_terms']}")
    print(f"invalid_characters={report['invalid_characters']}")


if __name__ == "__main__":
    main()

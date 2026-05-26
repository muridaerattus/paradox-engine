import argparse
import json
from pathlib import Path

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH, GENERATED_DIR, write_json
from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    build_dspy_examples,
    build_ranked_classpect_report,
    character_name,
    dataset_fingerprint,
    load_description_records,
    parse_ranked_labels,
    RankedClasspectRow,
    VALID_ASPECTS,
    VALID_CLASSES,
)
from experiments.classpect.scripts.generation.generate_descriptions import (
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    invoke_openai_compatible_chat,
)


def ranked_report_path(run_name: str) -> Path:
    safe_name = run_name.strip().replace(" ", "_")
    return GENERATED_DIR / f"ranked_classpect_report_{safe_name}.json"


def ranked_messages(personality_description: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Rank Homestuck class and aspect candidates from personality evidence only. "
                "Return only compact JSON with keys classes, aspects, abstain. "
                "classes and aspects must each contain up to three labels in descending likelihood. "
                "Use abstain true only if the evidence is too ambiguous. /no_think\n"
                "Valid classes: "
                + ", ".join(VALID_CLASSES)
                + "\nValid aspects: "
                + ", ".join(VALID_ASPECTS)
            ),
        },
        {
            "role": "user",
            "content": personality_description,
        },
    ]


def parse_abstained(raw: str) -> bool:
    lowered = raw.casefold()
    return '"abstain": true' in lowered or "abstain: true" in lowered


def evaluate_ranked_examples(
    *,
    descriptions_path: Path,
    run_name: str,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    request_timeout: int,
    resume: bool,
) -> None:
    examples = build_dspy_examples(load_description_records(descriptions_path))
    fingerprint = dataset_fingerprint(examples)
    output_path = ranked_report_path(run_name)
    rows: list[RankedClasspectRow] = []
    completed: set[tuple[int | None, int | None]] = set()
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        rows = [RankedClasspectRow.model_validate(row) for row in existing.get("rows", [])]
        completed = {(row.character_id, row.variant) for row in rows}
        print(f"resuming with existing_rows={len(rows)}", flush=True)

    total = len(examples)
    for index, example in enumerate(examples, start=1):
        key = (getattr(example, "character_id", None), getattr(example, "variant", None))
        if key in completed:
            continue
        print(f"evaluating ranked {index}/{total}: {character_name(example)}", flush=True)
        raw = ""
        error = None
        try:
            raw = invoke_openai_compatible_chat(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=ranked_messages(example.personality_description),
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            ranked_classes = parse_ranked_labels(raw, VALID_CLASSES)
            ranked_aspects = parse_ranked_labels(raw, VALID_ASPECTS)
            abstained = parse_abstained(raw)
        except Exception as exc:
            ranked_classes = []
            ranked_aspects = []
            abstained = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            RankedClasspectRow(
                character=character_name(example),
                character_id=getattr(example, "character_id", None),
                variant=getattr(example, "variant", None),
                expected_class=example.title_class,
                expected_aspect=example.title_aspect,
                ranked_classes=ranked_classes,
                ranked_aspects=ranked_aspects,
                abstained=abstained,
                raw_prediction=raw,
                prediction_error=error,
            )
        )
        report = build_ranked_classpect_report(rows, fingerprint=fingerprint)
        write_json(output_path, report.model_dump())

    report = build_ranked_classpect_report(rows, fingerprint=fingerprint)
    write_json(output_path, report.model_dump())
    print(f"report_path={output_path}", flush=True)
    print(f"row_count={report.row_count}", flush=True)
    print(f"class_top_1_accuracy={report.class_top_1_accuracy:.3f}", flush=True)
    print(f"class_top_2_accuracy={report.class_top_2_accuracy:.3f}", flush=True)
    print(f"class_top_3_accuracy={report.class_top_3_accuracy:.3f}", flush=True)
    print(f"aspect_top_1_accuracy={report.aspect_top_1_accuracy:.3f}", flush=True)
    print(f"aspect_top_2_accuracy={report.aspect_top_2_accuracy:.3f}", flush=True)
    print(f"aspect_top_3_accuracy={report.aspect_top_3_accuracy:.3f}", flush=True)
    print(f"title_top_1_accuracy={report.title_top_1_accuracy:.3f}", flush=True)
    print(f"title_top_2_accuracy={report.title_top_2_accuracy:.3f}", flush=True)
    print(f"title_top_3_accuracy={report.title_top_3_accuracy:.3f}", flush=True)
    print(f"abstention_rate={report.abstention_rate:.3f}", flush=True)
    print(f"prediction_error_rate={report.prediction_error_rate:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ranked top-k classpect candidates on descriptions."
    )
    parser.add_argument("--descriptions-path", type=Path, default=DESCRIPTIONS_PATH)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--api-base", default=DEFAULT_DSPY_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_DSPY_API_KEY)
    parser.add_argument("--model", default=DEFAULT_OPENAI_COMPATIBLE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    evaluate_ranked_examples(
        descriptions_path=args.descriptions_path,
        run_name=args.run_name,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

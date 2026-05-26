import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH, GENERATED_DIR, write_json
from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    VALID_ASPECTS,
    VALID_CLASSES,
    build_dspy_examples,
    character_name,
    dataset_fingerprint,
    load_description_records,
)
from experiments.classpect.scripts.generation.generate_descriptions import (
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    invoke_openai_compatible_chat,
)

PairwiseTarget = Literal["class", "aspect"]
PairwiseOutcome = Literal["true_label", "sink_label", "tie", "abstain", "error"]

DEFAULT_PAIR_SPECS: tuple[str, ...] = (
    "aspect:Breath:Hope",
    "aspect:Light:Mind",
    "aspect:Space:Life",
    "aspect:Space:Light",
    "aspect:Time:Doom",
    "class:Heir:Knight",
    "class:Page:Knight",
    "class:Seer:Mage",
    "class:Thief:Prince",
    "class:Thief:Knight",
)


class PairwiseSpec(BaseModel):
    target: PairwiseTarget
    label_a: str
    label_b: str


class PairwiseAuditRow(BaseModel):
    target: PairwiseTarget
    label_a: str
    label_b: str
    character: str
    character_id: int | None = None
    variant: int | None = None
    expected_label: str
    selected_label: str | None = None
    choice: str | None = None
    outcome: PairwiseOutcome
    raw_prediction: str = ""
    prediction_error: str | None = None


class PairwisePairSummary(BaseModel):
    target: PairwiseTarget
    label_a: str
    label_b: str
    row_count: int
    true_label_win_count: int
    true_label_win_rate: float
    sink_label_win_count: int
    sink_label_win_rate: float
    tie_count: int
    tie_rate: float
    abstention_count: int
    abstention_rate: float
    prediction_error_count: int
    prediction_error_rate: float


class PairwiseTargetSummary(BaseModel):
    target: PairwiseTarget
    row_count: int
    true_label_win_count: int
    true_label_win_rate: float
    sink_label_win_count: int
    sink_label_win_rate: float
    tie_count: int
    tie_rate: float
    abstention_count: int
    abstention_rate: float
    prediction_error_count: int
    prediction_error_rate: float


class PairwiseAuditReport(BaseModel):
    dataset_fingerprint: str
    row_count: int
    pair_summaries: list[PairwisePairSummary]
    target_summaries: list[PairwiseTargetSummary]
    rows: list[PairwiseAuditRow]


def pairwise_report_path(run_name: str) -> Path:
    safe_name = run_name.strip().replace(" ", "_")
    return GENERATED_DIR / f"pairwise_classpect_report_{safe_name}.json"


def parse_pair_spec(value: str) -> PairwiseSpec:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Pair specs must use target:label_a:label_b")
    target, label_a, label_b = parts
    if target not in {"class", "aspect"}:
        raise ValueError(f"Unknown pairwise target: {target}")
    choices = VALID_CLASSES if target == "class" else VALID_ASPECTS
    for label in (label_a, label_b):
        if label not in choices:
            raise ValueError(f"Invalid {target} label in pair spec: {label}")
    if label_a == label_b:
        raise ValueError("Pair labels must differ")
    return PairwiseSpec(target=target, label_a=label_a, label_b=label_b)


def pairwise_messages(
    *,
    target: PairwiseTarget,
    label_a: str,
    label_b: str,
    personality_description: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Choose between two Homestuck "
                f"{target} candidates using personality evidence only. "
                "Do not infer from names, aliases, handles, species, blood color, "
                "gender, dream moon, weapons, powers, canonical title labels, or plot metadata. "
                "Return exactly one token: A, B, Tie, or Abstain. /no_think"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Candidate A: {label_a}\n"
                f"Candidate B: {label_b}\n\n"
                "Anonymous personality description:\n"
                f"{personality_description}"
            ),
        },
    ]


def parse_pairwise_choice(raw: str) -> str:
    text = raw.strip()
    exact = text.casefold()
    if exact in {"a", "b", "tie", "abstain"}:
        return "A" if exact == "a" else "B" if exact == "b" else exact.title()

    matches = re.findall(r"\b(abstain|tie|a|b)\b", text, flags=re.IGNORECASE)
    normalized = {
        "A" if match.casefold() == "a" else "B" if match.casefold() == "b" else match.title()
        for match in matches
    }
    if len(normalized) == 1:
        return normalized.pop()
    raise ValueError(f"Expected exactly one of A, B, Tie, Abstain; got {raw!r}")


def expected_label(example: dspy.Example, target: PairwiseTarget) -> str:
    return example.title_class if target == "class" else example.title_aspect


def matching_examples(
    examples: Sequence[dspy.Example], spec: PairwiseSpec
) -> list[dspy.Example]:
    pair_labels = {spec.label_a, spec.label_b}
    return [example for example in examples if expected_label(example, spec.target) in pair_labels]


def build_pairwise_row(
    *, spec: PairwiseSpec, example: dspy.Example, raw: str, error: str | None
) -> PairwiseAuditRow:
    expected = expected_label(example, spec.target)
    selected_label = None
    choice = None
    outcome: PairwiseOutcome = "error"
    prediction_error = error
    if prediction_error is None:
        try:
            choice = parse_pairwise_choice(raw)
            if choice == "A":
                selected_label = spec.label_a
            elif choice == "B":
                selected_label = spec.label_b
            elif choice == "Tie":
                outcome = "tie"
            elif choice == "Abstain":
                outcome = "abstain"

            if selected_label is not None:
                outcome = "true_label" if selected_label == expected else "sink_label"
        except ValueError as exc:
            prediction_error = str(exc)

    return PairwiseAuditRow(
        target=spec.target,
        label_a=spec.label_a,
        label_b=spec.label_b,
        character=character_name(example),
        character_id=getattr(example, "character_id", None),
        variant=getattr(example, "variant", None),
        expected_label=expected,
        selected_label=selected_label,
        choice=choice,
        outcome=outcome,
        raw_prediction=raw,
        prediction_error=prediction_error,
    )


def summarize_pairwise_rows(
    rows: Sequence[PairwiseAuditRow], *, target: PairwiseTarget | None = None
) -> dict[str, int | float]:
    selected_rows = [row for row in rows if target is None or row.target == target]
    count = len(selected_rows) or 1
    true_label_win_count = sum(row.outcome == "true_label" for row in selected_rows)
    sink_label_win_count = sum(row.outcome == "sink_label" for row in selected_rows)
    tie_count = sum(row.outcome == "tie" for row in selected_rows)
    abstention_count = sum(row.outcome == "abstain" for row in selected_rows)
    prediction_error_count = sum(row.prediction_error is not None for row in selected_rows)
    return {
        "row_count": len(selected_rows),
        "true_label_win_count": true_label_win_count,
        "true_label_win_rate": true_label_win_count / count,
        "sink_label_win_count": sink_label_win_count,
        "sink_label_win_rate": sink_label_win_count / count,
        "tie_count": tie_count,
        "tie_rate": tie_count / count,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_count / count,
        "prediction_error_count": prediction_error_count,
        "prediction_error_rate": prediction_error_count / count,
    }


def build_pairwise_report(
    rows: Sequence[PairwiseAuditRow], specs: Sequence[PairwiseSpec], *, fingerprint: str
) -> PairwiseAuditReport:
    pair_summaries = []
    for spec in specs:
        pair_rows = [
            row
            for row in rows
            if row.target == spec.target
            and row.label_a == spec.label_a
            and row.label_b == spec.label_b
        ]
        pair_summaries.append(
            PairwisePairSummary(
                target=spec.target,
                label_a=spec.label_a,
                label_b=spec.label_b,
                **summarize_pairwise_rows(pair_rows),
            )
        )

    target_summaries = [
        PairwiseTargetSummary(
            target=target,
            **summarize_pairwise_rows(rows, target=target),
        )
        for target in ("class", "aspect")
    ]
    return PairwiseAuditReport(
        dataset_fingerprint=fingerprint,
        row_count=len(rows),
        pair_summaries=pair_summaries,
        target_summaries=target_summaries,
        rows=list(rows),
    )


def completed_key(row: PairwiseAuditRow) -> tuple[str, str, str, int | None, int | None]:
    return (row.target, row.label_a, row.label_b, row.character_id, row.variant)


def evaluate_pairwise_examples(
    *,
    descriptions_path: Path,
    run_name: str,
    pair_specs: Sequence[str],
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    request_timeout: int,
    resume: bool,
) -> None:
    specs = [parse_pair_spec(pair_spec) for pair_spec in pair_specs]
    examples = build_dspy_examples(load_description_records(descriptions_path))
    fingerprint = dataset_fingerprint(examples)
    output_path = pairwise_report_path(run_name)
    rows: list[PairwiseAuditRow] = []
    completed: set[tuple[str, str, str, int | None, int | None]] = set()
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        rows = [PairwiseAuditRow.model_validate(row) for row in existing.get("rows", [])]
        completed = {completed_key(row) for row in rows}
        print(f"resuming with existing_rows={len(rows)}", flush=True)

    tasks = [(spec, example) for spec in specs for example in matching_examples(examples, spec)]
    total = len(tasks)
    for index, (spec, example) in enumerate(tasks, start=1):
        key = (spec.target, spec.label_a, spec.label_b, example.character_id, example.variant)
        if key in completed:
            continue
        print(
            "evaluating pairwise "
            f"{index}/{total}: {spec.target} {spec.label_a} vs {spec.label_b} "
            f"for {character_name(example)}",
            flush=True,
        )
        raw = ""
        error = None
        try:
            raw = invoke_openai_compatible_chat(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=pairwise_messages(
                    target=spec.target,
                    label_a=spec.label_a,
                    label_b=spec.label_b,
                    personality_description=example.personality_description,
                ),
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append(build_pairwise_row(spec=spec, example=example, raw=raw, error=error))
        report = build_pairwise_report(rows, specs, fingerprint=fingerprint)
        write_json(output_path, report.model_dump())

    report = build_pairwise_report(rows, specs, fingerprint=fingerprint)
    write_json(output_path, report.model_dump())
    print(f"report_path={output_path}", flush=True)
    print(f"row_count={report.row_count}", flush=True)
    for summary in report.target_summaries:
        print(
            f"{summary.target}_true_label_win_rate={summary.true_label_win_rate:.3f}",
            flush=True,
        )
        print(
            f"{summary.target}_sink_label_win_rate={summary.sink_label_win_rate:.3f}",
            flush=True,
        )
        print(f"{summary.target}_tie_rate={summary.tie_rate:.3f}", flush=True)
        print(
            f"{summary.target}_abstention_rate={summary.abstention_rate:.3f}",
            flush=True,
        )
        print(
            f"{summary.target}_prediction_error_rate={summary.prediction_error_rate:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise classpect confusions on descriptions."
    )
    parser.add_argument("--descriptions-path", type=Path, default=DESCRIPTIONS_PATH)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pair", action="append", dest="pair_specs")
    parser.add_argument("--api-base", default=DEFAULT_DSPY_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_DSPY_API_KEY)
    parser.add_argument("--model", default=DEFAULT_OPENAI_COMPATIBLE_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    evaluate_pairwise_examples(
        descriptions_path=args.descriptions_path,
        run_name=args.run_name,
        pair_specs=args.pair_specs or DEFAULT_PAIR_SPECS,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

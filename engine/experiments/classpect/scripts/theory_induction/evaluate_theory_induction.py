import argparse
import json
from pathlib import Path

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH, write_json
from experiments.classpect.dspy_classpect import (
    DEFAULT_DSPY_API_BASE,
    DEFAULT_DSPY_API_KEY,
    build_dspy_examples,
    dataset_fingerprint,
    load_description_records,
    split_examples_grouped_by_fold,
)
from experiments.classpect.scripts.generation.generate_descriptions import (
    DEFAULT_OPENAI_COMPATIBLE_MODEL,
    invoke_openai_compatible_chat,
)
from experiments.classpect.theory_induction import (
    FoldTheorySet,
    InducedLabelTheory,
    TheoryScoredPrediction,
    build_theory_induction_diagnostics,
    build_theory_induction_report,
    build_theory_prediction,
    character_ids,
    contrast_examples_for_label,
    fold_theories_path,
    format_theory_examples,
    labels_for_theory_target,
    load_fold_theory_set,
    parse_induced_theory,
    source_examples_for_label,
    theory_context,
    theory_induction_diagnostics_path,
    theory_induction_report_path,
    write_fold_theory_set,
)


def theory_induction_messages(
    *,
    target: str,
    label: str,
    positive_examples: str,
    contrast_examples: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Induce a fold-local Homestuck classpect label interpretation from "
                "the provided canon-labeled training descriptions only. Do not use "
                "external analyst theories, names, aliases, handles, species, blood "
                "color, caste, gender, dream moon, lusus, planet, weapons, powers, "
                "canonical title labels beyond the supplied training label, or direct "
                "plot identifiers. Return only compact JSON with keys "
                "common_evidence, contrastive_evidence, uncertainty_notes. Each key "
                "must contain exactly three short strings of 18 words or fewer. "
                "Do not include markdown or commentary outside the JSON. /no_think"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target type: {target}\n"
                f"Target label: {label}\n\n"
                "Positive training examples for this label:\n"
                f"{positive_examples}\n\n"
                "Contrast training examples from other labels:\n"
                f"{contrast_examples}"
            ),
        },
    ]


def theory_scoring_messages(
    *,
    target: str,
    label_theories: str,
    personality_description: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Score an anonymous character description against fold-local "
                f"Homestuck {target} label theories. Use only the supplied theories "
                "and description, not external classpect theory or canon knowledge. "
                "Return only compact JSON with keys scores and abstain. scores must "
                "be a list with every label, each containing label, score, evidence, "
                "and rationale. score must be 0 to 5. evidence and rationale must "
                "each be 10 words or fewer. Use abstain true if the supplied theories "
                "do not support a meaningful distinction. Do not include markdown or "
                "commentary outside the JSON. /no_think"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fold-local {target} theories:\n"
                f"{label_theories}\n\n"
                "Anonymous personality description to score:\n"
                f"{personality_description}"
            ),
        },
    ]


def generate_fold_theories(
    *,
    variant: str,
    fold: int,
    folds: int,
    fingerprint: str,
    trainset,
    devset,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    request_timeout: int,
) -> FoldTheorySet:
    theories: list[InducedLabelTheory] = []
    for target in ("class", "aspect"):
        for label in labels_for_theory_target(target):
            positive_examples = source_examples_for_label(trainset, target, label)
            contrast_examples = contrast_examples_for_label(trainset, target, label)
            print(f"inducing theory fold={fold} {target}={label}", flush=True)
            raw = invoke_openai_compatible_chat(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=theory_induction_messages(
                    target=target,
                    label=label,
                    positive_examples=format_theory_examples(positive_examples, target),
                    contrast_examples=format_theory_examples(contrast_examples, target),
                ),
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            theories.append(
                parse_induced_theory(
                    raw,
                    target=target,
                    label=label,
                    source_character_count=len(positive_examples),
                    source_description_count=len(positive_examples),
                )
            )
    return FoldTheorySet(
        variant=variant,
        fold=fold,
        folds_total=folds,
        dataset_fingerprint=fingerprint,
        train_character_ids=character_ids(trainset),
        dev_character_ids=character_ids(devset),
        theories=sorted(theories, key=lambda theory: (theory.target, theory.label)),
    )


def completed_key(row: TheoryScoredPrediction) -> tuple[int, int | None, int | None]:
    return (row.fold, row.character_id, row.variant)


def evaluate_theory_induction(
    *,
    descriptions_path: Path,
    variant: str,
    folds: int,
    api_base: str,
    api_key: str,
    model: str,
    max_theory_tokens: int,
    max_score_tokens: int,
    request_timeout: int,
    resume: bool,
    retry_errors: bool,
) -> None:
    examples = build_dspy_examples(load_description_records(descriptions_path))
    fingerprint = dataset_fingerprint(examples)
    output_path = theory_induction_report_path(variant)
    diagnostics_path = theory_induction_diagnostics_path(variant)
    rows: list[TheoryScoredPrediction] = []
    completed: set[tuple[int, int | None, int | None]] = set()
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        rows = [TheoryScoredPrediction.model_validate(row) for row in existing.get("rows", [])]
        if retry_errors:
            rows = [row for row in rows if row.prediction_error is None]
        completed = {completed_key(row) for row in rows}
        print(f"resuming with existing_rows={len(rows)}", flush=True)

    for fold in range(folds):
        print(f"starting fold={fold}", flush=True)
        trainset, devset = split_examples_grouped_by_fold(examples, folds, fold)
        theory_path = fold_theories_path(variant, fold)
        if resume and theory_path.exists():
            theory_set = load_fold_theory_set(theory_path)
            print(f"loaded existing theories for fold={fold}", flush=True)
        else:
            theory_set = generate_fold_theories(
                variant=variant,
                fold=fold,
                folds=folds,
                fingerprint=fingerprint,
                trainset=trainset,
                devset=devset,
                api_base=api_base,
                api_key=api_key,
                model=model,
                max_tokens=max_theory_tokens,
                request_timeout=request_timeout,
            )
            write_fold_theory_set(theory_path, theory_set)
            print(f"wrote {theory_path}", flush=True)

        class_context = theory_context(theory_set.theories, "class")
        aspect_context = theory_context(theory_set.theories, "aspect")
        for index, example in enumerate(devset, start=1):
            key = (fold, example.character_id, example.variant)
            if key in completed:
                continue
            print(f"scoring fold={fold} row={index}/{len(devset)}", flush=True)
            raw_class = ""
            raw_aspect = ""
            class_error = None
            aspect_error = None
            try:
                raw_class = invoke_openai_compatible_chat(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=theory_scoring_messages(
                        target="class",
                        label_theories=class_context,
                        personality_description=example.personality_description,
                    ),
                    max_tokens=max_score_tokens,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                class_error = f"class: {type(exc).__name__}: {exc}"
            try:
                raw_aspect = invoke_openai_compatible_chat(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=theory_scoring_messages(
                        target="aspect",
                        label_theories=aspect_context,
                        personality_description=example.personality_description,
                    ),
                    max_tokens=max_score_tokens,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                aspect_error = f"aspect: {type(exc).__name__}: {exc}"
            rows.append(
                build_theory_prediction(
                    fold=fold,
                    example=example,
                    raw_class_prediction=raw_class,
                    raw_aspect_prediction=raw_aspect,
                    class_error=class_error,
                    aspect_error=aspect_error,
                )
            )
            report = build_theory_induction_report(
                variant=variant, fingerprint=fingerprint, folds=folds, rows=rows
            )
            write_json(output_path, report.model_dump())
            diagnostics = build_theory_induction_diagnostics(rows)
            write_json(diagnostics_path, diagnostics.model_dump())

    report = build_theory_induction_report(
        variant=variant, fingerprint=fingerprint, folds=folds, rows=rows
    )
    write_json(output_path, report.model_dump())
    diagnostics = build_theory_induction_diagnostics(rows)
    write_json(diagnostics_path, diagnostics.model_dump())
    print(f"report_path={output_path}", flush=True)
    print(f"diagnostics_path={diagnostics_path}", flush=True)
    print(f"row_count={report.row_count}", flush=True)
    print(f"class_accuracy={report.class_accuracy:.3f}", flush=True)
    print(f"aspect_accuracy={report.aspect_accuracy:.3f}", flush=True)
    print(f"title_accuracy={report.title_accuracy:.3f}", flush=True)
    print(f"class_top_3_accuracy={report.class_top_3_accuracy:.3f}", flush=True)
    print(f"aspect_top_3_accuracy={report.aspect_top_3_accuracy:.3f}", flush=True)
    print(f"title_top_3_accuracy={report.title_top_3_accuracy:.3f}", flush=True)
    print(f"prediction_error_rate={report.prediction_error_rate:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fold-local contrastive theory induction for classpect."
    )
    parser.add_argument("--descriptions-path", type=Path, default=DESCRIPTIONS_PATH)
    parser.add_argument("--variant", default="personality_only")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--api-base", default=DEFAULT_DSPY_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_DSPY_API_KEY)
    parser.add_argument("--model", default=DEFAULT_OPENAI_COMPATIBLE_MODEL)
    parser.add_argument("--max-theory-tokens", type=int, default=768)
    parser.add_argument("--max-score-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, discard rows with prediction_error and rescore them.",
    )
    args = parser.parse_args()

    evaluate_theory_induction(
        descriptions_path=args.descriptions_path,
        variant=args.variant,
        folds=args.folds,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_theory_tokens=args.max_theory_tokens,
        max_score_tokens=args.max_score_tokens,
        request_timeout=args.request_timeout,
        resume=args.resume,
        retry_errors=args.retry_errors,
    )


if __name__ == "__main__":
    main()

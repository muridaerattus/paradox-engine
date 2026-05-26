import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from experiments.classpect.decision_tree import DESCRIPTIONS_PATH, GENERATED_DIR, write_json
from experiments.classpect.training_examples import TRAINING_EXAMPLES, TrainingExample


DEFAULT_ANTHROPIC_DESCRIPTION_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "Qwen3.6-35B-A3B-MTP-GGUF"
DEFAULT_OPENAI_COMPATIBLE_API_BASE = os.environ.get("LOCAL_LLM_API_BASE", "http://localhost:8000/v1")
DEFAULT_OPENAI_COMPATIBLE_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "")


FORBIDDEN_METADATA_TERMS = (
    "blood color",
    "blood colour",
    "caste",
    "species",
    "troll",
    "human",
    "gender",
    "dream moon",
    "prospit",
    "derse",
    "lusus",
    "planet",
    "weapon",
    "strife",
    "powers",
    "canonical class",
    "canonical aspect",
    "title",
    "classpect",
    "protagonist",
    "antagonist",
    "main character",
    "supporting character",
    "side character",
    "villain",
    "hero",
)

KNOWN_ALIASES = (
    "ectoBiologist",
    "tentacleTherapist",
    "turntechGodhead",
    "gardenGnostic",
    "gutsyGumshoe",
    "tipsyGnostalgic",
    "timaeusTestified",
    "golgothasTerror",
    "carcinoGeneticist",
    "grimAuxiliatrix",
    "gallowsCalibrator",
    "arachnidsGrip",
    "terminallyCapricious",
    "caligulasAquarium",
    "cuttlefishCuller",
)


DESCRIPTION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You write compact personality descriptions for classpecting training data.

Describe the subject's personality, values, coping strategies, social behavior,
decision-making, conflict style, core insecurity, relationship to agency, social
role, and growth pattern. Refer to the subject only as "this person" or "they".

Do not include the character's name, aliases, handles, initials, typing quirks,
species, blood color, caste, gender, dream moon, lusus, planet, weapon, powers,
canonical class, canonical aspect, title, or direct plot identifiers. Do not rely
on those facts internally. The description must stand alone as an anonymous
personality profile.

The output should be 3-5 paragraphs and useful for later extraction of stable,
general personality features.
""".strip(),
        ),
        (
            "user",
            "Character: {character}\nKnown group for context only: {group}",
        ),
    ]
)


CONTRASTIVE_DESCRIPTION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You write compact anonymous personality descriptions for classpecting training data.

Describe the subject's personality, values, coping strategies, social behavior,
decision-making, conflict style, core insecurity, relationship to agency, social
role, and growth pattern. Refer to the subject only as "this person" or "they".

Prioritize contrastive evidence: include concrete personality signals that would
help distinguish this person from characters with nearby but different patterns.
Do not name or mention class labels, aspect labels, or contrast categories in the
output. The output should read as a natural anonymous personality profile, not as
a checklist.

For class-like distinctions, emphasize whether the person tends toward passive
adaptation, defensive service, interpretive guidance, personally embodied
understanding, acquisitive self-assertion, destructive rejection, self-creation
through maintenance, latent growth, redistributive support, or direct protective
duty when those patterns are relevant.

For aspect-like distinctions, emphasize whether the person orients around
knowledge and visibility, decision logic and consequences, freedom and
detachment, conviction and belief, environment and creation, vitality and
recovery, limits and costs, or hiddenness and absence when those patterns are
relevant.

Do not include the character's name, aliases, handles, initials, typing quirks,
species, blood color, caste, gender, dream moon, lusus, planet, weapon, powers,
canonical class, canonical aspect, title, or direct plot identifiers. Do not rely
on those facts internally. The description must stand alone as an anonymous
personality profile.

The output should be 3-5 paragraphs and useful for later classification from
personality evidence.
""".strip(),
        ),
        (
            "user",
            "Character: {character}\nKnown group for context only: {group}",
        ),
    ]
)


ASPECT_BALANCED_DESCRIPTION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You write compact anonymous personality descriptions for classpecting training data.

Describe the subject's personality, values, coping strategies, social behavior,
conflict style, core insecurity, relationship to agency, social role, and growth
pattern. Refer to the subject only as "this person" or "they".

Prioritize contrastive evidence, but keep aspect-like evidence balanced. Do not
let the description over-focus on analysis, judgment, decision logic, planning,
or consequences unless those are truly central. Those terms can make many people
look too similar. Include other kinds of motivation and pressure when relevant:
freedom versus attachment, urgency versus patience, vitality versus depletion,
visibility versus hiddenness, belief versus doubt, limits versus possibility,
and emotional interiority versus social obligation.

For class-like distinctions, describe how this person handles agency and social
function: passive adaptation, defensive service, interpretive guidance,
personally embodied understanding, acquisitive self-assertion, destructive
rejection, self-creation through maintenance, latent growth, redistributive
support, or direct protective duty. Use only patterns that fit.

For aspect-like distinctions, preserve evidence for under-recovered patterns:
independence, detachment, movement, and release; timing, pressure, repetition,
patience, and inevitability; placement, environment, creation, scale, and
spaciousness; emotional identity, inner conflict, personal authenticity, and
fragmented selfhood. Also include knowledge, choice, belief, vitality, limits,
or hiddenness only when they are specifically important.

Do not name or mention class labels, aspect labels, or contrast categories in the
output. Do not include the character's name, aliases, handles, initials, typing
quirks, species, blood color, caste, gender, dream moon, lusus, planet, weapon,
powers, canonical class, canonical aspect, title, or direct plot identifiers. Do
not rely on those facts internally. The description must stand alone as an
anonymous personality profile.

The output should be 3-5 paragraphs and useful for later classification from
personality evidence.
""".strip(),
        ),
        (
            "user",
            "Character: {character}\nKnown group for context only: {group}",
        ),
    ]
)


PERSONALITY_PLUS_ABSTRACT_ROLE_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You write anonymous classpecting dossiers for training data.

Describe the subject's personality and abstract narrative function. Include values,
coping strategies, social behavior, decision-making, conflict style, core
insecurity, relationship to agency, relationship to groups, recurring pressure,
growth pattern, and how they tend to affect the direction of a story.

The narrative-function evidence must stay abstract. Use functional phrasing like
"often catalyzes movement while avoiding direct responsibility" or "creates
pressure by forcing others to confront avoided commitments". Do not use role
labels such as protagonist, antagonist, hero, villain, main character, side
character, or supporting character.

Do not include the character's name, aliases, handles, initials, typing quirks,
species, blood color, caste, gender, dream moon, lusus, planet, weapon, powers,
canonical class, canonical aspect, title, classpect, or direct plot identifiers.
Do not mention specific events, locations, acts, sessions, deaths, transformations,
relationships by name, or setting-specific terminology. Refer to the subject only
as "this person" or "they".

The output should be 4-6 compact paragraphs. It should be useful for inducing
classpect interpretations from personality plus abstract story-role evidence,
while remaining anonymous and non-identifying.
""".strip(),
        ),
        (
            "user",
            "Character: {character}\nKnown group for context only: {group}",
        ),
    ]
)


def description_prompt(style: str) -> ChatPromptTemplate:
    if style == "baseline":
        return DESCRIPTION_PROMPT
    if style == "contrast-v2":
        return CONTRASTIVE_DESCRIPTION_PROMPT
    if style == "contrast-v3":
        return ASPECT_BALANCED_DESCRIPTION_PROMPT
    if style == "personality-plus-abstract-role":
        return PERSONALITY_PLUS_ABSTRACT_ROLE_PROMPT
    raise ValueError(f"Unknown description prompt style: {style}")


def forbidden_name_terms(example: TrainingExample) -> tuple[str, ...]:
    own_name_parts = tuple(part for part in example.character.split() if len(part) > 1)
    roster_name_parts = tuple(
        part
        for training_example in TRAINING_EXAMPLES
        for part in training_example.character.split()
        if len(part) > 2
    )
    return own_name_parts + roster_name_parts + KNOWN_ALIASES


def validate_description(description: str, example: TrainingExample) -> list[str]:
    errors = []
    for term in forbidden_name_terms(example):
        if re.search(rf"\b{re.escape(term)}\b", description, flags=re.IGNORECASE):
            errors.append(f"forbidden identifying term: {term}")
    lowered = description.casefold()
    for term in FORBIDDEN_METADATA_TERMS:
        if term in lowered:
            errors.append(f"forbidden metadata term: {term}")
    return errors


def rejected_terms(validation_errors: list[str]) -> list[str]:
    terms = []
    for err in validation_errors:
        term = err.split(":", 1)[-1].strip() if ":" in err else err.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def formatted_messages(
    prompt: ChatPromptTemplate,
    example: TrainingExample,
    validation_errors: list[str] | None = None,
) -> list[dict[str, str]]:
    role_by_type = {"human": "user", "ai": "assistant"}
    messages = [
        {
            "role": role_by_type.get(message.type, message.type),
            "content": str(message.content),
        }
        for message in prompt.format_messages(
            character=example.character, group=example.group
        )
    ]
    terms = rejected_terms(validation_errors or [])
    if terms:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous draft was rejected because it used these exact "
                    f"forbidden strings: {', '.join(terms)}. Regenerate from "
                    "scratch as an anonymous personality profile. Do not use those "
                    "strings, names, metadata labels, direct identifiers, or synonyms "
                    "for species, tools, powers, title, or setting facts."
                ),
            }
        )
    return messages


def invoke_openai_compatible_chat(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    request_timeout: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    chat_url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    chat_request = request.Request(chat_url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(chat_request, timeout=request_timeout) as response:
            data: dict[str, Any] = json.loads(response.read().decode())
    except error.HTTPError as http_error:
        body = http_error.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: {body}") from http_error
    if "choices" not in data:
        raise RuntimeError(f"OpenAI-compatible response missing choices: {data}")
    return str(data["choices"][0]["message"].get("content") or "")


async def generate_description(
    llm: ChatAnthropic | None,
    semaphore: asyncio.Semaphore,
    example: TrainingExample,
    variant: int,
    max_retries: int,
    prompt: ChatPromptTemplate,
    *,
    backend: str,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    request_timeout: int,
) -> dict:
    last_description = ""
    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        async with semaphore:
            if backend == "anthropic":
                if llm is None:
                    raise ValueError("Anthropic backend requires a ChatAnthropic client")
                if last_errors:
                    response = await llm.ainvoke(
                        formatted_messages(prompt, example, last_errors)
                    )
                else:
                    response = await (prompt | llm).ainvoke(
                        {"character": example.character, "group": example.group}
                    )
                last_description = response.content
            else:
                last_description = await asyncio.to_thread(
                    invoke_openai_compatible_chat,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=formatted_messages(prompt, example, last_errors),
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                )
        last_errors = validate_description(last_description, example)
        if not last_errors:
            break
        if attempt < max_retries:
            print(
                f"retrying {example.character} variant {variant}: {last_errors}",
                flush=True,
            )
    return {
        "character": example.character,
        "variant": variant,
        "description": last_description,
        "validation_errors": last_errors,
    }


async def generate_descriptions(
    model: str,
    concurrency: int,
    variants: int,
    max_retries: int,
    output_path: Path,
    prompt_style: str,
    *,
    backend: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    request_timeout: int,
    resume: bool,
    retry_invalid: bool,
) -> None:
    llm = ChatAnthropic(model=model, max_tokens=max_tokens) if backend == "anthropic" else None
    semaphore = asyncio.Semaphore(concurrency)
    prompt = description_prompt(prompt_style)
    records = []
    completed_keys: set[tuple[str, int]] = set()
    if resume and output_path.exists():
        records = json.loads(output_path.read_text())
        if retry_invalid:
            records = [record for record in records if not record.get("validation_errors")]
        completed_keys = {
            (str(record["character"]), int(record.get("variant", 0)))
            for record in records
        }
        print(f"resuming with existing_records={len(records)}", flush=True)
    tasks = [
        generate_description(
                llm,
                semaphore,
                example,
                variant,
                max_retries,
                prompt,
                backend=backend,
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            for example in TRAINING_EXAMPLES
            for variant in range(variants)
            if (example.character, variant) not in completed_keys
    ]
    total = len(records) + len(tasks)
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        write_json(output_path, records)
        print(
            f"completed {len(records)}/{total}: {record['character']} variant {record['variant']}",
            flush=True,
        )
    if not tasks:
        write_json(output_path, records)
    print(f"descriptions_path={output_path}", flush=True)


def main() -> None:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description="Generate gitignored personality descriptions for classpect examples."
    )
    parser.add_argument(
        "--backend",
        choices=("anthropic", "openai-compatible"),
        default="anthropic",
        help="LLM backend used for description generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model used for description generation. Defaults depend on backend.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_OPENAI_COMPATIBLE_API_BASE,
        help="OpenAI-compatible API base URL for local generation.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_OPENAI_COMPATIBLE_API_KEY,
        help="OpenAI-compatible API key for local generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens per description generation call.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds for OpenAI-compatible generation calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially written description dataset.",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="When resuming, discard invalid records so they are regenerated.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent LLM calls.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Independent descriptions to generate per character.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for descriptions that include forbidden identifying terms.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DESCRIPTIONS_PATH,
        help="Path for generated descriptions JSON.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Write descriptions_<run-name>.json under the generated directory.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=(
            "baseline",
            "contrast-v2",
            "contrast-v3",
            "personality-plus-abstract-role",
        ),
        default="baseline",
        help="Description prompt style to use.",
    )
    args = parser.parse_args()
    model = args.model
    if model is None:
        if args.backend == "anthropic":
            model = DEFAULT_ANTHROPIC_DESCRIPTION_MODEL
        else:
            model = DEFAULT_OPENAI_COMPATIBLE_MODEL
    output_path = args.output_path
    if args.run_name:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.run_name.strip()).strip("_")
        output_path = GENERATED_DIR / f"descriptions_{safe_name}.json"
    asyncio.run(
        generate_descriptions(
            model,
            args.concurrency,
            args.variants,
            args.max_retries,
            output_path,
            args.prompt_style,
            backend=args.backend,
            api_base=args.api_base,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            resume=args.resume,
            retry_invalid=args.retry_invalid,
        )
    )


if __name__ == "__main__":
    main()

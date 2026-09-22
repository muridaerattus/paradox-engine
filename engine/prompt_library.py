"""
Centralized loader for all prompt markdown files.
"""

import os
from typing import Literal, TypeAlias

from openrouter.components import (
    ChatSystemMessageTypedDict,
    ChatUserMessageTypedDict,
)

from settings import PROMPTS_DIRECTORY

ChatMessage: TypeAlias = ChatSystemMessageTypedDict | ChatUserMessageTypedDict

STRUCTURED_OUTPUT_INSTRUCTION = (
    "Return a JSON object that matches the response schema supplied with the request."
)


def _read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _read_dir(subdir: str) -> dict[str, str]:
    """Load every ``.md`` file in ``{PROMPTS_DIRECTORY}/{subdir}`` keyed by stem."""
    base = f"{PROMPTS_DIRECTORY}/{subdir}"
    return {
        os.path.splitext(name)[0]: _read_text(f"{base}/{name}")
        for name in os.listdir(base)
        if name.endswith(".md")
    }


# Class-of-Aspect summaries. Used by classpect (final narrative) and could be
# used by fraymotifs in the future if it wants class context.
CLASS_PROMPTS: dict[str, str] = _read_dir("classes")

# Aspect summaries. Used by classpect (final narrative) and fraymotifs
# (aspect_context per title).
ASPECT_PROMPTS: dict[str, str] = _read_dir("aspects")


QUIZ_ANSWERER_PROMPT_TEXT: str = _read_text(f"{PROMPTS_DIRECTORY}/quiz_answerer.md")
CLASS_EXAMPLE: str = _read_text(f"{PROMPTS_DIRECTORY}/class_example.md")
ASPECT_EXAMPLE: str = _read_text(f"{PROMPTS_DIRECTORY}/aspect_example.md")
PARADOX_ENGINE_PROMPT: str = _read_text(f"{PROMPTS_DIRECTORY}/paradox_engine.md")

def build_quiz_answerer_messages(
    *, character_description: str, questions: str, example: str
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": QUIZ_ANSWERER_PROMPT_TEXT.format(
                character_description=character_description,
                example=example,
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION,
            ),
        },
        {"role": "user", "content": f"QUESTIONS:\n{questions}"},
    ]

FRAYMOTIF_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/fraymotifs/fraymotif_generator.md"
)

def build_fraymotif_messages(
    *, players: str, memory: str, additional_info: str, aspect_context: str
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": FRAYMOTIF_GENERATOR_PROMPT.format(
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION
            ),
        },
        {
            "role": "user",
            "content": (
                f"Player Titles: {players}\n"
                f"Memory: {memory}\n"
                f"Additional Info: {additional_info}\n"
                f"Context for each aspect: {aspect_context}\n"
            ),
        },
    ]


ITEM_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_generator.md"
)
ITEM_DESCRIPTION_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_description_generator.md"
)
ITEM_TAGLINE_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_tagline_generator.md"
)

def item_generator_messages(
    *,
    item_1_name: str,
    item_1_components: str,
    item_1_description: str,
    item_2_name: str,
    item_2_components: str,
    item_2_description: str,
    operation: Literal["and", "or"],
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": ITEM_GENERATOR_PROMPT.format(
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION
            ),
        },
        {
            "role": "user",
            "content": (
                f"Item 1: {item_1_name}\n"
                f"Item 1 Components: {item_1_components}\n"
                f"Item 1 Description: {item_1_description}\n\n"
                f"Item 2: {item_2_name}\n"
                f"Item 2 Components: {item_2_components}\n"
                f"Item 2 Description: {item_2_description}\n"
                f"Operation: {operation}"
            ),
        },
    ]

def build_item_description_messages(*, name: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": ITEM_DESCRIPTION_GENERATOR_PROMPT},
        {"role": "user", "content": name},
    ]


def build_item_tagline_messages(*, name: str, description: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": ITEM_TAGLINE_GENERATOR_PROMPT},
        {"role": "user", "content": f"{name}\n{description}"},
    ]


def build_paradox_engine_messages(
    *,
    character_description: str,
    class_data: str,
    aspect_data: str,
    title: str,
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": PARADOX_ENGINE_PROMPT.format(
                character_description=character_description,
                class_data=class_data,
                aspect_data=aspect_data,
            ),
        },
        {"role": "user", "content": title},
    ]

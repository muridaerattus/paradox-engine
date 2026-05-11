"""
Centralized loader for all prompt markdown files.
"""

import os

from langchain_core.prompts import ChatPromptTemplate

from settings import PROMPTS_DIRECTORY


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

QUIZ_ANSWERER_CHAT_PROMPT = ChatPromptTemplate(
    [
        ("system", QUIZ_ANSWERER_PROMPT_TEXT),
        ("user", "QUESTIONS:\n{questions}"),
    ]
)

FRAYMOTIF_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/fraymotifs/fraymotif_generator.md"
)

FRAYMOTIF_CHAT_PROMPT = ChatPromptTemplate(
    [
        ("system", FRAYMOTIF_GENERATOR_PROMPT),
        (
            "user",
            """
         Player Titles: {players}
         Memory: {memory}
         Additional Info: {additional_info}
         Context for each aspect: {aspect_context}""",
        ),
    ]
)


ITEM_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_generator.md"
)
ITEM_DESCRIPTION_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_description_generator.md"
)
ITEM_TAGLINE_GENERATOR_PROMPT: str = _read_text(
    f"{PROMPTS_DIRECTORY}/alchemy/item_tagline_generator.md"
)

ITEM_GENERATOR_CHAT_PROMPT = ChatPromptTemplate(
    [
        ("system", ITEM_GENERATOR_PROMPT),
        (
            "user",
            """
         Item 1: {item_1_name}
         Item 1 Components: {item_1_components}
         Item 1 Description: {item_1_description}
         
         Item 2: {item_2_name}
         Item 2 Components: {item_2_components}
         Item 2 Description: {item_2_description}
         Operation: {operation}""",
        ),
    ]
)
ITEM_DESCRIPTION_CHAT_PROMPT = ChatPromptTemplate(
    [("system", ITEM_DESCRIPTION_GENERATOR_PROMPT), ("user", "{name}")]
)
ITEM_TAGLINE_CHAT_PROMPT = ChatPromptTemplate(
    [
        ("system", ITEM_TAGLINE_GENERATOR_PROMPT),
        ("user", "{name}\n{description}"),
    ]
)

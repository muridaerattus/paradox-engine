from pathlib import Path

from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import (
    fraymotif_messages,
    item_generator_messages,
    quiz_answerer_messages,
)


def prompts() -> PromptLibrary:
    return PromptLibrary.load(Path("prompts"))


def test_quiz_prompt_interpolates_all_placeholders():
    messages = quiz_answerer_messages(
        prompts(),
        character_description="A test character",
        questions="1. A question?",
        example="An example",
    )
    assert "A test character" in messages[0]["content"]
    assert "An example" in messages[0]["content"]
    assert "{format_instructions}" not in messages[0]["content"]
    assert messages[1]["content"] == "QUESTIONS:\n1. A question?"


def test_structured_prompt_builders_remove_format_placeholder():
    library = prompts()
    fraymotif = fraymotif_messages(
        library,
        players="Mage of Time",
        memory="A memory",
        additional_info="Some details",
        aspect_context="Time context",
    )
    item = item_generator_messages(
        library,
        item_1_name="Hammer",
        item_1_components="Hammer",
        item_1_description="A hammer",
        item_2_name="Clock",
        item_2_components="Clock",
        item_2_description="A clock",
        operation="and",
    )
    assert "{format_instructions}" not in fraymotif[0]["content"]
    assert "{format_instructions}" not in item[0]["content"]

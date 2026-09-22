from typing import Literal, TypeAlias

from openrouter.components import (
    ChatAssistantMessageTypedDict,
    ChatSystemMessageTypedDict,
    ChatUserMessageTypedDict,
)

from paradox_engine.prompts.library import PromptLibrary

ChatMessage: TypeAlias = (
    ChatSystemMessageTypedDict
    | ChatUserMessageTypedDict
    | ChatAssistantMessageTypedDict
)
STRUCTURED_OUTPUT_INSTRUCTION = (
    "Return a JSON object that matches the response schema supplied with the request."
)


def classpect_interview_messages(
    prompts: PromptLibrary,
    *,
    history: list[dict],
    unresolved_questions: list[str],
) -> list[ChatMessage]:
    persona = prompts.paradox_engine.split("<TASK>", maxsplit=1)[0]
    if unresolved_questions:
        coverage_instruction = (
            "The classifier still lacks sufficient evidence for these quiz items:\n- "
            + "\n- ".join(unresolved_questions)
            + "\nAsk one question that will best resolve this missing evidence."
        )
    else:
        coverage_instruction = (
            "The classifier can confidently answer every class and aspect quiz item. "
            "Do not ask another question; produce the personality summary."
        )
    system_message = {
        "role": "system",
        "content": persona
        + prompts.classpect_interviewer.format(
            coverage_instruction=coverage_instruction,
            format_instructions=STRUCTURED_OUTPUT_INSTRUCTION,
        ),
    }
    return [system_message, *history]


def quiz_answerer_messages(
    prompts: PromptLibrary, *, character_description: str, questions: str, example: str
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": prompts.quiz_answerer.format(
                character_description=character_description,
                example=example,
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION,
            ),
        },
        {"role": "user", "content": f"QUESTIONS:\n{questions}"},
    ]


def fraymotif_messages(
    prompts: PromptLibrary,
    *,
    players: str,
    memory: str,
    additional_info: str,
    aspect_context: str,
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": prompts.fraymotif_generator.format(
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION
            ),
        },
        {
            "role": "user",
            "content": (
                f"Player Titles: {players}\nMemory: {memory}\n"
                f"Additional Info: {additional_info}\n"
                f"Context for each aspect: {aspect_context}\n"
            ),
        },
    ]


def item_generator_messages(
    prompts: PromptLibrary,
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
            "content": prompts.item_generator.format(
                format_instructions=STRUCTURED_OUTPUT_INSTRUCTION
            ),
        },
        {
            "role": "user",
            "content": (
                f"Item 1: {item_1_name}\nItem 1 Components: {item_1_components}\n"
                f"Item 1 Description: {item_1_description}\n\n"
                f"Item 2: {item_2_name}\nItem 2 Components: {item_2_components}\n"
                f"Item 2 Description: {item_2_description}\nOperation: {operation}"
            ),
        },
    ]


def item_description_messages(
    prompts: PromptLibrary, *, name: str
) -> list[ChatMessage]:
    return [
        {"role": "system", "content": prompts.item_description_generator},
        {"role": "user", "content": name},
    ]


def item_tagline_messages(
    prompts: PromptLibrary, *, name: str, description: str
) -> list[ChatMessage]:
    return [
        {"role": "system", "content": prompts.item_tagline_generator},
        {"role": "user", "content": f"{name}\n{description}"},
    ]


def paradox_engine_messages(
    prompts: PromptLibrary,
    *,
    character_description: str,
    class_data: str,
    aspect_data: str,
    title: str,
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": prompts.paradox_engine.format(
                character_description=character_description,
                class_data=class_data,
                aspect_data=aspect_data,
            ),
        },
        {"role": "user", "content": title},
    ]

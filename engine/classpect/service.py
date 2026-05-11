import asyncio
import logging
import random

from pydantic import ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from classpect.models import ParadoxEngineOutput
from classpect.utils import format_answer_string, quiz_to_model, generate_question_list
from prompt_library import (
    ASPECT_EXAMPLE,
    ASPECT_PROMPTS,
    CLASS_EXAMPLE,
    CLASS_PROMPTS,
    PARADOX_ENGINE_PROMPT,
    QUIZ_ANSWERER_CHAT_PROMPT,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def answer_questions(
    quiz_json: dict,
    llm,
    prompt: ChatPromptTemplate,
    character_description: str,
    example: str,
) -> str:
    results = set()
    for question in quiz_json:
        for answer in question["answers"]:
            answer["answer"] = await format_answer_string(answer["answer"])
            results.update(answer["personality_types"])

    quiz_model = await quiz_to_model(quiz_json)
    question_list = await generate_question_list(quiz_json)

    structured_llm = llm.with_structured_output(quiz_model, include_raw=True)
    parser = PydanticOutputParser(pydantic_object=quiz_model)
    prompted_llm = prompt | structured_llm
    raw_result = await prompted_llm.ainvoke(
        {
            "character_description": character_description,
            "format_instructions": parser.get_format_instructions(),
            "questions": question_list,
            "example": example,
        }
    )

    parsing_error = (
        raw_result.get("parsing_error") if isinstance(raw_result, dict) else None
    )
    llm_response = (
        raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
    )
    if parsing_error is not None or llm_response is None:
        raw_message = (
            raw_result.get("raw") if isinstance(raw_result, dict) else raw_result
        )
        logger.error(
            "Quiz answerer failed to produce a valid structured response. "
            "parsing_error=%r raw=%r",
            parsing_error,
            raw_message,
        )
        if isinstance(parsing_error, BaseException):
            raise parsing_error
        raise ValidationError.from_exception_data("QuizAnswers", [])

    logger.info(llm_response.ThinkingSpace)

    # Iterate quiz_model fields in declaration order, skipping ThinkingSpace
    answers_in_order = [
        getattr(llm_response, name).value
        for name in quiz_model.model_fields
        if name != "ThinkingSpace"
    ]
    logger.info(answers_in_order)

    result_scores = {result: 0 for result in results}
    for i, question in enumerate(quiz_json):
        answer_list = question["answers"]
        answers_by_text = {a["answer"]: a["personality_types"] for a in answer_list}

        answer = answers_in_order[i]
        personality_types = answers_by_text[answer]
        for result in personality_types:
            result_scores[result] += 1
        logger.info(f"{question['question']}: {answer}")

    logger.info(result_scores)

    max_score = max(result_scores.values())
    max_results = [res for res in result_scores if result_scores[res] == max_score]
    logger.info(max_results)
    return random.choice(max_results)


async def calculate_title(
    character_description: str, class_quiz_json: dict, aspect_quiz_json: dict
) -> ParadoxEngineOutput:
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=8192)

    # Class and aspect quizzes are independent, run them concurrently
    class_result, aspect_result = await asyncio.gather(
        answer_questions(
            class_quiz_json,
            llm,
            QUIZ_ANSWERER_CHAT_PROMPT,
            character_description,
            CLASS_EXAMPLE,
        ),
        answer_questions(
            aspect_quiz_json,
            llm,
            QUIZ_ANSWERER_CHAT_PROMPT,
            character_description,
            ASPECT_EXAMPLE,
        ),
    )

    class_result = class_result.split(" ")[0].capitalize()
    aspect_result = aspect_result.capitalize()

    logger.info(f"Final answer: {class_result} of {aspect_result}")

    class_prompt = CLASS_PROMPTS[class_result.lower()]
    aspect_prompt = ASPECT_PROMPTS[aspect_result.lower()]

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=8192)
    prompt = ChatPromptTemplate(
        [
            ("system", PARADOX_ENGINE_PROMPT),
            ("user", f"{class_result} of {aspect_result}"),
        ]
    )
    prompted_llm = prompt | llm
    llm_response = await prompted_llm.ainvoke(
        {
            "character_description": character_description,
            "class_data": class_prompt,
            "aspect_data": aspect_prompt,
        }
    )

    return ParadoxEngineOutput(
        class_result=class_result,
        aspect_result=aspect_result,
        llm_response=llm_response.content,
    )

import asyncio
import logging
import random

from pydantic import ValidationError

from classpect.models import ParadoxEngineOutput
from classpect.utils import format_answer_string, quiz_to_model, generate_question_list
from llm import generate_structured, generate_text
from prompt_library import (
    ASPECT_EXAMPLE,
    ASPECT_PROMPTS,
    build_paradox_engine_messages,
    build_quiz_answerer_messages,
    CLASS_EXAMPLE,
    CLASS_PROMPTS,
)
from settings import CLASSPECT_MODEL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def answer_questions(
    quiz_json: dict,
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

    try:
        llm_response = await generate_structured(
            model=CLASSPECT_MODEL,
            messages=build_quiz_answerer_messages(
                character_description=character_description,
                questions=question_list,
                example=example,
            ),
            output_type=quiz_model,
            max_tokens=8192,
        )
    except ValidationError:
        logger.exception("Quiz answerer failed to produce a valid response")
        raise

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
    # Class and aspect quizzes are independent, run them concurrently
    class_result, aspect_result = await asyncio.gather(
        answer_questions(
            class_quiz_json,
            character_description,
            CLASS_EXAMPLE,
        ),
        answer_questions(
            aspect_quiz_json,
            character_description,
            ASPECT_EXAMPLE,
        ),
    )

    class_result = class_result.split(" ")[0].capitalize()
    aspect_result = aspect_result.capitalize()

    logger.info(f"Final answer: {class_result} of {aspect_result}")

    class_prompt = CLASS_PROMPTS[class_result.lower()]
    aspect_prompt = ASPECT_PROMPTS[aspect_result.lower()]

    title = f"{class_result} of {aspect_result}"
    llm_response = await generate_text(
        model=CLASSPECT_MODEL,
        messages=build_paradox_engine_messages(
            character_description=character_description,
            class_data=class_prompt,
            aspect_data=aspect_prompt,
            title=title,
        ),
        max_tokens=8192,
    )

    return ParadoxEngineOutput(
        class_result=class_result,
        aspect_result=aspect_result,
        llm_response=llm_response,
    )

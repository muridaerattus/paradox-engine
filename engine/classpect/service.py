import asyncio
import logging
import random

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


async def answer_questions(quiz_json, llm, prompt, character_description, example):
    results = set()
    for question in quiz_json:
        for answer in question["answers"]:
            answer["answer"] = await format_answer_string(answer["answer"])
            results.update(answer["personality_types"])

    quiz_model = await quiz_to_model(quiz_json)
    question_list = await generate_question_list(quiz_json)

    structured_llm = llm.with_structured_output(quiz_model)
    parser = PydanticOutputParser(pydantic_object=quiz_model)
    prompted_llm = prompt | structured_llm
    llm_response = await prompted_llm.ainvoke(
        {
            "character_description": character_description,
            "format_instructions": parser.get_format_instructions(),
            "questions": question_list,
            "example": example,
        }
    )
    logger.info(llm_response.ThinkingSpace)

    answer_objects = [x for x in llm_response][1:]
    answers_in_order = [x[1]._value_ for x in answer_objects if not isinstance(x, str)]
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
    character_description, class_quiz_json, aspect_quiz_json
) -> ParadoxEngineOutput:
    llm = ChatAnthropic(model="claude-sonnet-4-6")

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

    llm = ChatAnthropic(model="claude-sonnet-4-6")
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

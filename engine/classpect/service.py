import aiofiles
import random
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from classpect.models import ParadoxEngineOutput
from classpect.utils import format_answer_string, quiz_to_model, generate_question_list


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
    # llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo") # Good for diverse results. May not follow proper formatting all the time.
    # llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo") # Better for diverse results, but a bit overkill.
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022") # Default; says "Rogue of Doom/Rage/Time" a lot.
    # llm = ChatAnthropic(model="claude-3-opus-latest") # Overkill.
    llm = ChatAnthropic(model="claude-3-7-sonnet-latest")  # Untested.
    async with aiofiles.open("prompts/quiz_answerer.md") as f:
        quiz_answerer_prompt_text = await f.read()
    quiz_answerer_prompt = ChatPromptTemplate(
        [("system", quiz_answerer_prompt_text), ("user", "QUESTIONS:\n{questions}")]
    )

    class_example = None
    aspect_example = None
    async with aiofiles.open("prompts/class_example.md") as f:
        class_example = await f.read()
    async with aiofiles.open("prompts/aspect_example.md") as f:
        aspect_example = await f.read()

    class_result = await answer_questions(
        class_quiz_json, llm, quiz_answerer_prompt, character_description, class_example
    )
    aspect_result = await answer_questions(
        aspect_quiz_json,
        llm,
        quiz_answerer_prompt,
        character_description,
        aspect_example,
    )

    class_result = class_result.split(" ")[0].capitalize()
    aspect_result = aspect_result.capitalize()

    logger.info(f"Final answer: {class_result} of {aspect_result}")

    # retrieve class and aspect summaries
    class_prompt = None
    aspect_prompt = None
    paradox_engine_prompt = None

    async with aiofiles.open(f"prompts/classes/{class_result.lower()}.md") as f:
        class_prompt = await f.read()
    async with aiofiles.open(f"prompts/aspects/{aspect_result.lower()}.md") as f:
        aspect_prompt = await f.read()
    async with aiofiles.open("prompts/paradox_engine.md") as f:
        paradox_engine_prompt = await f.read()

    llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
    # llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    prompt = ChatPromptTemplate(
        [
            ("system", paradox_engine_prompt),
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

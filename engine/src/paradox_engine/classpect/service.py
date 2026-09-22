import asyncio
import logging
import random

from pydantic import ValidationError

from paradox_engine.classpect.models import ParadoxEngineOutput
from paradox_engine.classpect.utils import (
    format_answer_string,
    generate_question_list,
    quiz_to_classifier_questions,
    quiz_to_model,
)
from paradox_engine.config import Settings
from paradox_engine.llm import LLMClient
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import (
    paradox_engine_messages,
    quiz_answerer_messages,
)

logger = logging.getLogger(__name__)


class ClasspectService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptLibrary,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.settings = settings

    @staticmethod
    def score_answers(quiz_json: list[dict], answer_indexes: list[int]) -> str:
        if len(answer_indexes) != len(quiz_json):
            raise ValueError("The number of quiz answers does not match the questions")

        results = {
            result
            for question in quiz_json
            for answer in question["answers"]
            for result in answer["personality_types"]
        }
        result_scores = {result: 0 for result in results}

        for question, answer_index in zip(quiz_json, answer_indexes):
            try:
                answer = question["answers"][answer_index]
            except IndexError as exc:
                raise ValueError("Quiz answer index is out of range") from exc
            for result in answer["personality_types"]:
                result_scores[result] += 1
            logger.info(f"{question['question']}: {answer['answer']}")

        logger.info(result_scores)
        max_score = max(result_scores.values())
        max_results = [
            result for result in result_scores if result_scores[result] == max_score
        ]
        logger.info(max_results)
        return random.choice(max_results)

    async def answer_questions_with_llm(
        self,
        quiz_json: dict,
        character_description: str,
        example: str,
    ) -> str:
        for question in quiz_json:
            for answer in question["answers"]:
                answer["answer"] = await format_answer_string(answer["answer"])

        quiz_model = await quiz_to_model(quiz_json)
        question_list = await generate_question_list(quiz_json)

        try:
            llm_response = await self.llm.generate_structured(
                model=self.settings.classpect_model,
                messages=quiz_answerer_messages(
                    self.prompts,
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

        answer_indexes = []
        for question, answer in zip(quiz_json, answers_in_order):
            answers_by_text = {
                item["answer"]: index for index, item in enumerate(question["answers"])
            }
            answer_indexes.append(answers_by_text[answer])
        return self.score_answers(quiz_json, answer_indexes)

    async def answer_questions_with_classifier(
        self, quiz_json: list[dict], character_description: str
    ) -> str:
        questions, answer_lookups = quiz_to_classifier_questions(quiz_json)
        choices = await self.llm.classify_choices(
            model=self.settings.classpect_classifier_model,
            state={"personality": character_description},
            questions=questions,
        )
        answer_indexes = [
            answer_lookups[question_name][choices[question_name]]
            for question_name in questions
        ]
        return self.score_answers(quiz_json, answer_indexes)

    async def answer_questions(
        self,
        quiz_json: dict,
        character_description: str,
        example: str,
    ) -> str:
        if self.settings.classpect_mode == "classifier":
            return await self.answer_questions_with_classifier(
                quiz_json, character_description
            )
        return await self.answer_questions_with_llm(
            quiz_json, character_description, example
        )

    async def calculate_title(
        self,
        character_description: str,
        class_quiz_json: dict,
        aspect_quiz_json: dict,
    ) -> ParadoxEngineOutput:
        # Class and aspect quizzes are independent, run them concurrently
        class_result, aspect_result = await asyncio.gather(
            self.answer_questions(
                class_quiz_json,
                character_description,
                self.prompts.class_example,
            ),
            self.answer_questions(
                aspect_quiz_json,
                character_description,
                self.prompts.aspect_example,
            ),
        )

        return await self.generate_title(
            character_description,
            class_result=class_result,
            aspect_result=aspect_result,
        )

    async def calculate_title_from_answers(
        self,
        character_description: str,
        class_quiz_json: list[dict],
        aspect_quiz_json: list[dict],
        class_answer_indexes: list[int],
        aspect_answer_indexes: list[int],
    ) -> ParadoxEngineOutput:
        class_result = self.score_answers(class_quiz_json, class_answer_indexes)
        aspect_result = self.score_answers(aspect_quiz_json, aspect_answer_indexes)
        return await self.generate_title(
            character_description,
            class_result=class_result,
            aspect_result=aspect_result,
        )

    async def generate_title(
        self,
        character_description: str,
        *,
        class_result: str,
        aspect_result: str,
    ) -> ParadoxEngineOutput:
        class_result = class_result.split(" ")[0].capitalize()
        aspect_result = aspect_result.capitalize()

        logger.info(f"Final answer: {class_result} of {aspect_result}")

        class_prompt = self.prompts.classes[class_result.lower()]
        aspect_prompt = self.prompts.aspects[aspect_result.lower()]

        title = f"{class_result} of {aspect_result}"
        llm_response = await self.llm.generate_text(
            model=self.settings.classpect_model,
            messages=paradox_engine_messages(
                self.prompts,
                character_description=character_description,
                class_data=class_prompt,
                aspect_data=aspect_prompt,
                title=title,
            ),
            max_tokens=8192,
            reasoning_effort="minimal",
        )

        return ParadoxEngineOutput(
            class_result=class_result,
            aspect_result=aspect_result,
            llm_response=llm_response,
        )

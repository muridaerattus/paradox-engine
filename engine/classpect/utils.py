from enum import Enum
from pydantic import create_model
from pydantic.fields import FieldInfo


async def format_answer_string(s: str):
    return s.lower().replace(".", "").replace('"', "")


async def quiz_to_model(quiz_json: dict):
    enums = [
        Enum(
            question["question"],
            ((answer["answer"], answer["answer"]) for answer in question["answers"]),
            type=str,
        )
        for question in quiz_json
    ]
    chain_of_thought_attrs = {
        "ThinkingSpace": (
            str,
            FieldInfo(
                description="Space to think about the general themes of the questions before you answer them, given your personality."
            ),
        )
    }
    answer_attrs = {
        f"Answer{i + 1}": (
            question_enum,
            FieldInfo(description=f'Answer to the question "{question_enum.__name__}"'),
        )
        for i, question_enum in enumerate(enums)
    }
    attrs = chain_of_thought_attrs | answer_attrs
    quiz_model = create_model("QuizAnswers", **attrs)
    return quiz_model


async def generate_question_list(quiz_json: dict):
    return "\n".join(
        [f"{i + 1}. {question['question']}" for i, question in enumerate(quiz_json)]
    )

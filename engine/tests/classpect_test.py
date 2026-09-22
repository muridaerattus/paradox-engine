import asyncio
from pathlib import Path

from paradox_engine.classpect.service import ClasspectService
from paradox_engine.classpect.utils import quiz_to_classifier_questions
from paradox_engine.config import Settings
from paradox_engine.prompts.library import PromptLibrary

QUIZ = [
    {
        "question": "Do you prefer groups or solitude?",
        "answers": [
            {"answer": "Groups", "personality_types": ["blood"]},
            {"answer": "Solitude", "personality_types": ["breath"]},
        ],
    },
    {
        "question": "Do you plan ahead?",
        "answers": [
            {"answer": "Yes", "personality_types": ["blood"]},
            {"answer": "No", "personality_types": ["breath"]},
        ],
    },
]


class FakeLLM:
    def __init__(self):
        self.request = {}

    async def classify_choices(self, **kwargs):
        self.request.update(kwargs)
        return {"question_1": "option_1", "question_2": "option_1"}


def make_service(mode="classifier"):
    return ClasspectService(
        llm=FakeLLM(),
        prompts=PromptLibrary.load(Path("prompts")),
        settings=Settings(classpect_mode=mode),
    )


def test_quiz_to_classifier_questions_uses_stable_keys():
    questions, lookups = quiz_to_classifier_questions(QUIZ)
    assert list(questions) == ["question_1", "question_2"]
    assert questions["question_1"]["type"] == "choice"
    assert questions["question_1"]["criteria"] == {
        "option_1": "Groups",
        "option_2": "Solitude",
    }
    assert lookups["question_1"] == {"option_1": 0, "option_2": 1}


def test_classifier_answers_are_scored():
    service = make_service()
    result = asyncio.run(
        service.answer_questions_with_classifier(QUIZ, "Community-oriented")
    )
    assert result == "blood"
    assert service.llm.request["state"] == {"personality": "Community-oriented"}
    assert list(service.llm.request["questions"]) == ["question_1", "question_2"]


def test_answer_questions_routes_by_mode(monkeypatch):
    calls = []
    service = make_service()

    async def classifier(*args):
        calls.append("classifier")
        return "blood"

    async def llm(*args):
        calls.append("llm")
        return "breath"

    monkeypatch.setattr(service, "answer_questions_with_classifier", classifier)
    monkeypatch.setattr(service, "answer_questions_with_llm", llm)
    assert (
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example"))
        == "blood"
    )

    service.settings.classpect_mode = "llm"
    assert (
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example"))
        == "breath"
    )
    assert calls == ["classifier", "llm"]

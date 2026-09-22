import asyncio

from classpect import service
from classpect.utils import quiz_to_classifier_questions


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


def test_quiz_to_classifier_questions_uses_stable_keys():
    questions, lookups = quiz_to_classifier_questions(QUIZ)

    assert list(questions) == ["question_1", "question_2"]
    assert questions["question_1"]["type"] == "choice"
    assert questions["question_1"]["criteria"] == {
        "option_1": "Groups",
        "option_2": "Solitude",
    }
    assert (
        "Do you prefer groups or solitude?" in questions["question_1"]["instructions"]
    )
    assert lookups["question_1"] == {"option_1": 0, "option_2": 1}


def test_classifier_answers_are_scored(monkeypatch):
    request = {}

    async def classify_choices(**kwargs):
        request.update(kwargs)
        return {"question_1": "option_1", "question_2": "option_1"}

    monkeypatch.setattr(service, "classify_choices", classify_choices)

    result = asyncio.run(
        service.answer_questions_with_classifier(QUIZ, "Community-oriented")
    )

    assert result == "blood"
    assert request["model"] == service.CLASSPECT_CLASSIFIER_MODEL
    assert request["state"] == {"personality": "Community-oriented"}
    assert list(request["questions"]) == ["question_1", "question_2"]


def test_answer_questions_routes_by_mode(monkeypatch):
    calls = []

    async def classifier(*args):
        calls.append("classifier")
        return "blood"

    async def llm(*args):
        calls.append("llm")
        return "breath"

    monkeypatch.setattr(service, "answer_questions_with_classifier", classifier)
    monkeypatch.setattr(service, "answer_questions_with_llm", llm)

    monkeypatch.setattr(service, "CLASSPECT_MODE", "classifier")
    assert (
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example")) == "blood"
    )

    monkeypatch.setattr(service, "CLASSPECT_MODE", "llm")
    assert (
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example"))
        == "breath"
    )
    assert calls == ["classifier", "llm"]

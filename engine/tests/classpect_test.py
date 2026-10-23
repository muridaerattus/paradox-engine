import asyncio
from copy import deepcopy
import logging
from pathlib import Path

import pytest

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


def make_service(mode="classifier", **settings):
    return ClasspectService(
        llm=FakeLLM(),
        prompts=PromptLibrary.load(Path("prompts")),
        settings=Settings(classpect_mode=mode, **settings),
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
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example")) == "blood"
    )

    service.settings.classpect_mode = "llm"
    assert (
        asyncio.run(service.answer_questions(QUIZ, "Personality", "Example"))
        == "breath"
    )
    assert calls == ["classifier", "llm"]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, True), ("yes", True), ("no", False)]
)
def test_llm_output_logging_setting_defaults_on_and_reads_environment(
    monkeypatch, value, expected
):
    monkeypatch.delenv("LOG_LLM_OUTPUTS", raising=False)
    if value is not None:
        monkeypatch.setenv("LOG_LLM_OUTPUTS", value)
    assert Settings(_env_file=None).log_llm_outputs is expected


@pytest.mark.parametrize("enabled", [True, False])
def test_llm_reasoning_and_answers_logging_is_configurable(
    monkeypatch, caplog, enabled
):
    service = make_service(mode="llm", log_llm_outputs=enabled)

    async def generate_structured(**kwargs):
        return kwargs["output_type"](
            ThinkingSpace="Private model reasoning", Answer1="groups", Answer2="yes"
        )

    monkeypatch.setattr(
        service.llm, "generate_structured", generate_structured, raising=False
    )
    with caplog.at_level(logging.INFO, logger="paradox_engine.classpect.service"):
        result = asyncio.run(
            service.answer_questions_with_llm(deepcopy(QUIZ), "Personality", "Example")
        )
    assert result == "blood"
    assert ("Private model reasoning" in caplog.text) is enabled
    assert ("['groups', 'yes']" in caplog.text) is enabled
    assert ("Do you prefer groups or solitude?: groups" in caplog.text) is enabled
    assert ("'blood': 2" in caplog.text) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_classifier_answer_logging_is_configurable(caplog, enabled):
    service = make_service(log_llm_outputs=enabled)
    with caplog.at_level(logging.INFO, logger="paradox_engine.classpect.service"):
        result = asyncio.run(
            service.answer_questions_with_classifier(QUIZ, "Personality")
        )
    assert result == "blood"
    assert ("Do you prefer groups or solitude?: Groups" in caplog.text) is enabled
    assert ("'blood': 2" in caplog.text) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_final_title_logging_is_configurable(monkeypatch, caplog, enabled):
    service = make_service(log_llm_outputs=enabled)

    async def generate_text(**kwargs):
        return "Your title explanation."

    monkeypatch.setattr(service.llm, "generate_text", generate_text, raising=False)
    with caplog.at_level(logging.INFO, logger="paradox_engine.classpect.service"):
        result = asyncio.run(
            service.generate_title(
                "Personality", class_result="seer", aspect_result="blood"
            )
        )
    assert result.llm_response == "Your title explanation."
    assert ("Final answer: Seer of Blood" in caplog.text) is enabled

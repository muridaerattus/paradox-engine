import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from paradox_engine.llm import LLMClient


class ExampleOutput(BaseModel):
    answer: str


def _response(content: str):
    message = SimpleNamespace(content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_generate_text(monkeypatch):
    llm = LLMClient(None)
    request = {}

    async def send_async(**kwargs):
        request.update(kwargs)
        return _response("hello")

    monkeypatch.setattr(llm.client.chat, "send_async", send_async)

    result = asyncio.run(
        llm.generate_text(
            model="test/model",
            messages=[{"role": "user", "content": "Say hello"}],
        )
    )

    assert result == "hello"
    assert request["model"] == "test/model"
    assert "response_format" not in request
    assert "reasoning_effort" not in request
    assert request["provider"].sort == "throughput"
    assert request["provider"].require_parameters is True


def test_generate_text_forwards_reasoning_effort(monkeypatch):
    llm = LLMClient(None)
    request = {}

    async def send_async(**kwargs):
        request.update(kwargs)
        return _response("hello")

    monkeypatch.setattr(llm.client.chat, "send_async", send_async)

    asyncio.run(
        llm.generate_text(
            model="z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="minimal",
        )
    )

    assert request["reasoning_effort"] == "minimal"


def test_generate_structured(monkeypatch):
    llm = LLMClient(None)
    request = {}

    async def send_async(**kwargs):
        request.update(kwargs)
        return _response('{"answer":"structured"}')

    monkeypatch.setattr(llm.client.chat, "send_async", send_async)

    result = asyncio.run(
        llm.generate_structured(
            model="test/model",
            messages=[{"role": "user", "content": "Answer"}],
            output_type=ExampleOutput,
        )
    )

    assert result == ExampleOutput(answer="structured")
    assert request["provider"].require_parameters is True
    assert request["provider"].sort == "throughput"

    response_format = request["response_format"].model_dump()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_classify_choices(monkeypatch):
    llm = LLMClient(None)
    request = {}

    async def create_async(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            answers={"question_1": SimpleNamespace(type="choice", choice="option_2")}
        )

    monkeypatch.setattr(llm.client.system_one, "create_async", create_async)
    questions = {
        "question_1": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"option_1": "One", "option_2": "Two"},
        }
    }

    result = asyncio.run(
        llm.classify_choices(
            model="jev-latest",
            state={"personality": "A test personality"},
            questions=questions,
        )
    )

    assert result == {"question_1": "option_2"}
    assert request == {
        "model": "jev-latest",
        "state": {"personality": "A test personality"},
        "questions": questions,
    }


def test_classify_choices_rejects_unknown_choice(monkeypatch):
    llm = LLMClient(None)
    async def create_async(**kwargs):
        return SimpleNamespace(
            answers={
                "question_1": SimpleNamespace(type="choice", choice="not-an-option")
            }
        )

    monkeypatch.setattr(llm.client.system_one, "create_async", create_async)

    try:
        asyncio.run(
            llm.classify_choices(
                model="jev-latest",
                state="Test",
                questions={
                    "question_1": {
                        "type": "choice",
                        "instructions": "Pick one",
                        "criteria": {"option_1": "One"},
                    }
                },
            )
        )
    except ValueError as exc:
        assert "unknown choice" in str(exc)
    else:
        raise AssertionError("Expected an unknown classifier choice to fail")


@pytest.mark.parametrize(
    ("answers", "error_type", "message"),
    [
        ({}, ValueError, "did not answer"),
        (
            {"question_1": SimpleNamespace(type="noul", noul=0.9)},
            TypeError,
            "non-choice",
        ),
    ],
)
def test_classify_choices_rejects_invalid_answers(
    monkeypatch, answers, error_type, message
):
    llm = LLMClient(None)
    async def create_async(**kwargs):
        return SimpleNamespace(answers=answers)

    monkeypatch.setattr(llm.client.system_one, "create_async", create_async)

    with pytest.raises(error_type, match=message):
        asyncio.run(
            llm.classify_choices(
                model="jev-latest",
                state="Test",
                questions={
                    "question_1": {
                        "type": "choice",
                        "instructions": "Pick one",
                        "criteria": {"option_1": "One"},
                    }
                },
            )
        )

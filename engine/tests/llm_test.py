import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

import llm


class ExampleOutput(BaseModel):
    answer: str


def _response(content: str):
    message = SimpleNamespace(content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_generate_text(monkeypatch):
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


def test_generate_structured(monkeypatch):
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

    response_format = request["response_format"].model_dump()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert (
        response_format["json_schema"]["schema"]["additionalProperties"] is False
    )

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from paradox_engine.api.body_limit import RequestBodyLimitMiddleware
from paradox_engine.app import create_app
from paradox_engine.config import Settings


def test_application_rejects_oversized_json_before_validation():
    app = create_app(Settings(_env_file=None, max_request_body_bytes=32))
    # No lifespan is needed: rejection must happen before services are accessed.
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.post("/classpect", json={"personality": "x" * 100})
    finally:
        client.close()
    assert response.status_code == 413


def test_body_at_limit_is_preserved():
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=8)

    @app.post("/")
    async def echo(request: Request):
        return {"body": (await request.body()).decode()}

    with TestClient(app) as client:
        response = client.post("/", content=b"12345678")
    assert response.status_code == 200
    assert response.json() == {"body": "12345678"}


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
def test_streamed_body_is_counted_even_with_missing_or_false_length(headers):
    called = False
    sent = []

    async def downstream(scope, receive, send):
        nonlocal called
        called = True

    messages = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56789", "more_body": False},
        ]
    )

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=8)
    asyncio.run(middleware({"type": "http", "headers": headers}, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_oversized_declared_length_does_not_read_body():
    async def unexpected(*args):
        pytest.fail("Body or downstream app must not be accessed")

    sent = []

    async def send(message):
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(unexpected, max_bytes=8)
    asyncio.run(
        middleware(
            {"type": "http", "headers": [(b"content-length", b"9")]}, unexpected, send
        )
    )
    assert sent[0]["status"] == 413

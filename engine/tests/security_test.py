from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paradox_engine.alchemy import repository
from paradox_engine.api.dependencies import resources
from paradox_engine.api.rate_limit import IPRateLimiter
from paradox_engine.api.routers import alchemy, classpect, fraymotifs
from paradox_engine.config import get_settings


def test_database_logging_respects_settings_and_hides_parameters():
    assert repository.engine.echo == get_settings().database_echo
    assert repository.engine.sync_engine.hide_parameters is True


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/classpect", {"personality": "Curious."}),
        (
            "/alchemy/alchemize",
            {"item_one": "hammer", "item_two": "clock", "operation": "and"},
        ),
        ("/fraymotif", {"players": "Seer of Light", "memory": "Discovery."}),
    ],
)
def test_generation_routes_share_a_limit_before_services_are_called(path, payload):
    runtime = SimpleNamespace(
        classpect=SimpleNamespace(
            calculate_title=AsyncMock(
                return_value=SimpleNamespace(
                    class_result="Seer",
                    aspect_result="Light",
                    llm_response="Your title.",
                )
            )
        ),
        class_quiz=[],
        aspect_quiz=[],
        alchemy=SimpleNamespace(
            alchemize_items=AsyncMock(
                return_value=SimpleNamespace(
                    name="Clockhammer",
                    code="abcdefgh",
                    description="A tool.",
                    tagline="!",
                )
            )
        ),
        fraymotifs=SimpleNamespace(
            create_fraymotif=AsyncMock(
                return_value=SimpleNamespace(
                    visual_description="Light.",
                    name="Discovery",
                    mechanical_description="",
                )
            )
        ),
    )
    app = FastAPI()
    app.state.rate_limiter = IPRateLimiter()
    for router in (alchemy.router, classpect.router, fraymotifs.router):
        app.include_router(router)
    app.dependency_overrides[resources] = lambda: runtime

    with TestClient(app) as client:
        assert client.post(path, json=payload).status_code == 200
        app.state.rate_limiter = IPRateLimiter()
        # A different endpoint consumes the shared cooldown without invoking an LLM.
        assert client.post("/classpect", json={"personality": ""}).status_code == 422
        response = client.post(
            path, json=payload, headers={"X-Forwarded-For": "192.0.2.99"}
        )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "15"
    assert runtime.classpect.calculate_title.await_count == (path == "/classpect")
    assert runtime.alchemy.alchemize_items.await_count == (path == "/alchemy/alchemize")
    assert runtime.fraymotifs.create_fraymotif.await_count == (path == "/fraymotif")

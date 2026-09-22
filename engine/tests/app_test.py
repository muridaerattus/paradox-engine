from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paradox_engine.app import create_app
from paradox_engine.config import Settings


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "prompts_directory": Path("prompts"),
        "class_quiz_filename": Path("class_quiz.json"),
        "aspect_quiz_filename": Path("aspect_quiz.json"),
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
    }
    values.update(overrides)
    return Settings(**values)


def test_health_endpoint(tmp_path: Path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "PARADOX ENGINE: Status operational."}


def test_startup_reports_missing_prompt_directory(tmp_path: Path):
    app = create_app(_settings(tmp_path, prompts_directory=tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="prompt directory is missing"):
        with TestClient(app):
            pass


def test_request_length_is_validated(tmp_path: Path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/classpect", json={"personality": ""})
    assert response.status_code == 422

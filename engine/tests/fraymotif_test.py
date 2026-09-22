from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paradox_engine.app import create_app
from paradox_engine.config import Settings
from paradox_engine.fraymotifs.models import Title
from paradox_engine.fraymotifs.utils import format_titles, split_titles


def test_split_titles():
    assert split_titles("Rogue of Doom") == (["rogue"], ["doom"])
    assert split_titles("Rogue of Doom, Mage of Time") == (
        ["rogue", "mage"],
        ["doom", "time"],
    )
    assert split_titles("") == ([], [])
    with pytest.raises(ValueError):
        split_titles("Rogue of Doom, Mage")


def test_format_titles():
    assert format_titles([Title(title_class="Rogue", title_aspect="Doom")]) == (
        "Player 1: Rogue of Doom"
    )
    assert format_titles([]) == ""


def test_invalid_title_returns_bad_request(tmp_path: Path):
    settings = Settings(
        prompts_directory=Path("prompts"),
        class_quiz_filename=Path("class_quiz.json"),
        aspect_quiz_filename=Path("aspect_quiz.json"),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/fraymotif",
            json={
                "players": "Not a title",
                "memory": "A memory",
                "additional_info": "None",
            },
        )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Titles must be in the format 'Class of Aspect'."
    }

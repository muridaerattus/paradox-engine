import pytest
from fastapi.testclient import TestClient
from fraymotifs.utils import split_titles, format_titles
from fraymotifs.models import Title
from main import app


def test_split_titles():
    # Test with a simple title
    titles = split_titles("Rogue of Doom")
    assert titles == (["rogue"], ["doom"])

    # Test with multiple titles
    titles = split_titles("Rogue of Doom, Mage of Time")
    assert titles == (["rogue", "mage"], ["doom", "time"])

    # Test with empty string
    titles = split_titles("")
    assert titles == ([], [])

    # Test with invalid format
    with pytest.raises(ValueError):
        split_titles("Rogue of Doom, Mage")


def test_format_titles():
    # Test with a single title
    titles = format_titles([Title(title_class="Rogue", title_aspect="Doom")])
    assert titles == "Player 1: Rogue of Doom"

    # Test with multiple titles
    titles = format_titles(
        [
            Title(title_class="Rogue", title_aspect="Doom"),
            Title(title_class="Mage", title_aspect="Time"),
        ]
    )
    assert titles == "Player 1: Rogue of Doom\nPlayer 2: Mage of Time"

    # Test with empty list
    titles = format_titles([])
    assert titles == ""


def test_invalid_title_returns_bad_request():
    with TestClient(app) as client:
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

from dataclasses import dataclass
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required prompt file is missing: {path}") from exc


def _read_directory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"Required prompt directory is missing: {path}")
    return {file.stem: _read_text(file) for file in sorted(path.glob("*.md"))}


@dataclass(frozen=True)
class PromptLibrary:
    classes: dict[str, str]
    aspects: dict[str, str]
    quiz_answerer: str
    class_example: str
    aspect_example: str
    paradox_engine: str
    fraymotif_generator: str
    item_generator: str
    item_description_generator: str
    item_tagline_generator: str

    @classmethod
    def load(cls, root: Path) -> "PromptLibrary":
        return cls(
            classes=_read_directory(root / "classes"),
            aspects=_read_directory(root / "aspects"),
            quiz_answerer=_read_text(root / "quiz_answerer.md"),
            class_example=_read_text(root / "class_example.md"),
            aspect_example=_read_text(root / "aspect_example.md"),
            paradox_engine=_read_text(root / "paradox_engine.md"),
            fraymotif_generator=_read_text(root / "fraymotifs/fraymotif_generator.md"),
            item_generator=_read_text(root / "alchemy/item_generator.md"),
            item_description_generator=_read_text(
                root / "alchemy/item_description_generator.md"
            ),
            item_tagline_generator=_read_text(root / "alchemy/item_tagline_generator.md"),
        )

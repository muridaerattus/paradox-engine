from fraymotifs.models import Title, Fraymotif
from fraymotifs.utils import format_titles
from llm import generate_structured

from prompt_library import ASPECT_PROMPTS, build_fraymotif_messages
from settings import FRAYMOTIF_MODEL


async def generate_aspect_context(titles: list[Title]) -> str:
    """
    Retrieve information about the aspects of the given titles.
    """
    if not titles:
        return ""
    total_aspect_info = ""

    for title in titles:
        if not title.title_aspect:
            raise ValueError("All titles must have an aspect defined.")
        aspect_info = ASPECT_PROMPTS[title.title_aspect.lower()]
        total_aspect_info += f"Aspect: {title.title_aspect}\nInfo: {aspect_info}\n\n"

    return total_aspect_info.strip()


async def create_fraymotif(
    titles: list[Title], memory: str, additional_info: str
) -> Fraymotif:
    """
    Creates a fraymotif based on the provided titles, memory, and additional information.
    """
    if not titles or not memory:
        raise ValueError("Titles and memory must be provided to create a fraymotif.")

    if len(titles) < 1:
        raise ValueError("At least one title is required to create a fraymotif.")

    aspect_context = await generate_aspect_context(titles)
    players_formatted = format_titles(titles)
    fraymotif = await generate_structured(
        model=FRAYMOTIF_MODEL,
        messages=build_fraymotif_messages(
            players=players_formatted,
            memory=memory,
            additional_info=additional_info,
            aspect_context=aspect_context,
        ),
        output_type=Fraymotif,
    )
    return fraymotif

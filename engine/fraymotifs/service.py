from fraymotifs.models import Title, Fraymotif
from fraymotifs.utils import format_titles
from settings import PROMPTS_DIRECTORY
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import aiofiles


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
        async with aiofiles.open(
            f"{PROMPTS_DIRECTORY}/aspects/{title.title_aspect.lower()}.md"
        ) as f:
            aspect_info = await f.read()
            total_aspect_info += (
                f"Aspect: {title.title_aspect}\nInfo: {aspect_info}\n\n"
            )

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

    async with aiofiles.open(
        f"{PROMPTS_DIRECTORY}/fraymotifs/fraymotif_generator.md"
    ) as f:
        fraymotif_prompt = await f.read()
    aspect_context = await generate_aspect_context(titles)
    prompt = ChatPromptTemplate(
        [
            ("system", fraymotif_prompt),
            (
                "user",
                """
         Player Titles: {players}
         Memory: {memory}
         Additional Info: {additional_info}
         Context for each aspect: {aspect_context}""",
            ),
        ]
    )
    llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
    parser = PydanticOutputParser(pydantic_object=Fraymotif)
    format_instructions = parser.get_format_instructions()
    llm_chain = prompt | llm | parser
    players_formatted = format_titles(titles)
    fraymotif: Fraymotif = await llm_chain.ainvoke(
        {
            "players": players_formatted,
            "memory": memory,
            "additional_info": additional_info,
            "aspect_context": aspect_context,
            "format_instructions": format_instructions,
        }
    )
    return fraymotif

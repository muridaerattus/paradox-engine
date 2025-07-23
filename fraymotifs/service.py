from fraymotifs.models import FraymotifInput, Title, Fraymotif
from fraymotifs.utils import format_titles
from settings import PROMPTS_DIRECTORY
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import aiofiles

async def create_fraymotif(titles: list[Title], memory: str, additional_info: str) -> Fraymotif:
    """
    Creates a fraymotif based on the provided titles, memory, and additional information.
    """
    if not titles or not memory:
        raise ValueError("Titles and memory must be provided to create a fraymotif.")
    
    if len(titles) < 1:
        raise ValueError("At least one title is required to create a fraymotif.")
    
    async with aiofiles.open(f"{PROMPTS_DIRECTORY}/fraymotif/fraymotif_generator.md") as f:
        fraymotif_prompt = await f.read()
    prompt = ChatPromptTemplate([
        ('system', fraymotif_prompt),
        ('user', """
         Players: {players}
         Memory: {memory}
         Additional Info: {additional_info}""")
    ])
    llm = ChatAnthropic(model="claude-3-7-sonnet-latest")
    parser = PydanticOutputParser(pydantic_object=Fraymotif)
    format_instructions = parser.get_format_instructions()
    llm_chain = prompt | llm | parser
    players_formatted = format_titles(titles)
    fraymotif: Fraymotif = await llm_chain.ainvoke({
        'players': players_formatted,
        'memory': memory,
        'additional_info': additional_info,
        'format_instructions': format_instructions
    })
    return fraymotif

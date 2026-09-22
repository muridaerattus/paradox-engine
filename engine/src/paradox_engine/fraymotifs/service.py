from paradox_engine.fraymotifs.models import Fraymotif, Title
from paradox_engine.fraymotifs.utils import format_titles
from paradox_engine.llm import LLMClient
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import fraymotif_messages


class FraymotifService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptLibrary,
        model: str,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.model = model

    async def generate_aspect_context(self, titles: list[Title]) -> str:
        """
        Retrieve information about the aspects of the given titles.
        """
        if not titles:
            return ""
        total_aspect_info = ""

        for title in titles:
            if not title.title_aspect:
                raise ValueError("All titles must have an aspect defined.")
            aspect_info = self.prompts.aspects[title.title_aspect.lower()]
            total_aspect_info += (
                f"Aspect: {title.title_aspect}\nInfo: {aspect_info}\n\n"
            )

        return total_aspect_info.strip()

    async def create_fraymotif(
        self, titles: list[Title], memory: str, additional_info: str
    ) -> Fraymotif:
        """
        Creates a fraymotif based on the provided titles, memory, and additional information.
        """
        if not titles or not memory:
            raise ValueError("Titles and memory must be provided to create a fraymotif.")

        if len(titles) < 1:
            raise ValueError("At least one title is required to create a fraymotif.")

        aspect_context = await self.generate_aspect_context(titles)
        players_formatted = format_titles(titles)
        fraymotif = await self.llm.generate_structured(
            model=self.model,
            messages=fraymotif_messages(
                self.prompts,
                players=players_formatted,
                memory=memory,
                additional_info=additional_info,
                aspect_context=aspect_context,
            ),
            output_type=Fraymotif,
        )
        return fraymotif

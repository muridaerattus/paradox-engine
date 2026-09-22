import asyncio
from typing import Protocol

from paradox_engine.alchemy.models import (
    Item,
    AlchemizedItem,
    Operation,
    generate_alchemy_code,
    format_name,
)
from paradox_engine.alchemy.operations import alchemy_and, alchemy_or
from paradox_engine.llm import LLMClient
from paradox_engine.prompts.library import PromptLibrary
from paradox_engine.prompts.messages import (
    item_description_messages,
    item_generator_messages,
    item_tagline_messages,
)


class AlchemyRepository(Protocol):
    async def get_item_by_name_or_code(self, name_or_code: str) -> Item | None: ...

    async def get_item_by_code(self, code: str) -> Item | None: ...

    async def insert_item(self, item: Item) -> Item: ...


class AlchemyService:
    def __init__(
        self,
        *,
        repository: AlchemyRepository,
        llm: LLMClient,
        prompts: PromptLibrary,
        model: str,
    ) -> None:
        self.repository = repository
        self.llm = llm
        self.prompts = prompts
        self.model = model

    async def alchemize_items(
        self, item_1_name: str, item_2_name: str, operation: Operation
    ) -> Item:
        """
        Combine two items using their alchemy codes.

        :param item_1: First item to combine
        :param item_2: Second item to combine
        :param operation: The operation to perform ('and' or 'or')
        :return: A new Item instance with a combined alchemy code
        """

        item_1, item_2 = await asyncio.gather(
            self.repository.get_item_by_name_or_code(item_1_name),
            self.repository.get_item_by_name_or_code(item_2_name),
        )

        if not item_1 and not item_2:
            if item_1_name == item_2_name:
                item_1 = await self.new_item(item_1_name)
                await self.repository.insert_item(item_1)
                item_2 = item_1  # Sorry to ruin the fun :)
            else:
                item_1, item_2 = await asyncio.gather(
                    self.new_item(item_1_name),
                    self.new_item(item_2_name),
                )
                await asyncio.gather(
                    self.repository.insert_item(item_1),
                    self.repository.insert_item(item_2),
                )
        elif not item_1:
            item_1 = await self.new_item(item_1_name)
            await self.repository.insert_item(item_1)
        elif not item_2:
            if item_1_name == item_2_name:
                item_2 = item_1  # Sorry to ruin the fun :)
            else:
                item_2 = await self.new_item(item_2_name)
                await self.repository.insert_item(item_2)

        combined_item = await self.generate_item(item_1, item_2, operation)
        await self.repository.insert_item(combined_item)

        return combined_item


    async def new_item(self, name: str) -> Item:
        """
        Create a new item with a unique alchemy code.
        Usually used for common objects that aren't in the database, like "hammer" or "needles".

        :param name: Name of the item
        :param description: Description of the item (optional)
        :return: An Item instance with a generated alchemy code
        """
        formatted_name = format_name(name)
        description = await self.generate_description(formatted_name)
        tagline = await self.generate_tagline(formatted_name, description)
        code = generate_alchemy_code()
        return Item(
            name=formatted_name,
            components=formatted_name,
            description=description or "",
            tagline=tagline or "",
            code=code,
        )


    async def generate_item(
        self, item_1: Item, item_2: Item, operation: Operation
    ) -> Item:
        """
        Generate a name for the new item based on the names of the two items being combined.

        :param name_1: Name of the first item
        :param name_2: Name of the second item
        :param operation: The operation used ('and' or 'or')
        :return: A generated name string
        """
        combined_code = None
        combined_components = None
        match operation:
            case "and":
                combined_code = alchemy_and(item_1.code, item_2.code)
                combined_components = f"({item_1.components}) && ({item_2.components})"
            case "or":
                combined_code = alchemy_or(item_1.code, item_2.code)
                combined_components = f"({item_1.components}) || ({item_2.components})"
            case _:
                raise ValueError(f"Invalid operation: {operation}. Use 'and' or 'or'.")

        existing_item = await self.repository.get_item_by_code(combined_code)
        if existing_item:
            return existing_item

        alchemized_item = await self.llm.generate_structured(
            model=self.model,
            messages=item_generator_messages(
                self.prompts,
                item_1_name=item_1.name,
                item_1_components=item_1.components,
                item_1_description=item_1.description,
                item_2_name=item_2.name,
                item_2_components=item_2.components,
                item_2_description=item_2.description,
                operation=operation,
            ),
            output_type=AlchemizedItem,
        )

        return Item(
            code=combined_code,
            components=combined_components,
            name=alchemized_item.name,
            description=alchemized_item.description,
            tagline=await self.generate_tagline(
                alchemized_item.name, alchemized_item.description
            ),
        )


    async def generate_description(self, name: str) -> str:
        """
        Generate a standalone description for the item based on its name.
        Used for when the item can't be found in the database.

        :param name: Name of the item
        :return: A generated description string
        """
        return await self.llm.generate_text(
            model=self.model,
            messages=item_description_messages(self.prompts, name=name),
        )


    async def generate_tagline(self, name: str, description: str) -> str:
        """
        Generate a tagline for the item based on its name and description.

        :param name: Name of the item
        :param description: Description of the item
        :return: A generated tagline string
        """
        return await self.llm.generate_text(
            model=self.model,
            messages=item_tagline_messages(
                self.prompts,
                name=name,
                description=description,
            ),
        )

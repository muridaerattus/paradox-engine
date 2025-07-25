from alchemy.models import (
    Item,
    ItemNameAndDescription,
    Operation,
    generate_alchemy_code,
    format_name,
)
from alchemy.operations import alchemy_and, alchemy_or
from database.alchemy_database import (
    get_item_by_name_or_code,
    get_item_by_code,
    insert_item,
)
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from settings import PROMPTS_DIRECTORY
import aiofiles


async def alchemize_items(
    item_1_name: str, item_2_name: str, operation: Operation
) -> Item:
    """
    Combine two items using their alchemy codes.

    :param item_1: First item to combine
    :param item_2: Second item to combine
    :param operation: The operation to perform ('and' or 'or')
    :return: A new Item instance with a combined alchemy code
    """

    item_1 = await get_item_by_name_or_code(item_1_name)
    item_2 = await get_item_by_name_or_code(item_2_name)

    if not item_1:
        item_1 = await new_item(item_1_name)
        await insert_item(item_1)
    if not item_2:
        item_2 = await new_item(item_2_name)
        await insert_item(item_2)

    combined_item = await generate_item(item_1, item_2, operation)
    await insert_item(combined_item)

    return combined_item


async def new_item(name: str) -> Item:
    """
    Create a new item with a unique alchemy code.
    Usually used for common objects that aren't in the database, like "hammer" or "needles".

    :param name: Name of the item
    :param description: Description of the item (optional)
    :return: An Item instance with a generated alchemy code
    """
    formatted_name = format_name(name)
    description = await generate_description(formatted_name)
    code = generate_alchemy_code()
    return Item(
        name=formatted_name,
        components=formatted_name,
        description=description or "",
        code=code,
    )


async def generate_item(item_1: Item, item_2: Item, operation: Operation) -> Item:
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

    existing_item = await get_item_by_code(combined_code)
    if existing_item:
        return existing_item

    item_name_prompt = None
    async with aiofiles.open(f"{PROMPTS_DIRECTORY}/alchemy/item_generator.md") as f:
        item_name_prompt = await f.read()

    prompt = ChatPromptTemplate(
        [
            ("system", item_name_prompt),
            (
                "user",
                """
         Item 1: {item_1_name}
         Item 1 Components: {item_1_components}
         Item 1 Description: {item_1_description}
         
         Item 2: {item_2_name}
         Item 2 Components: {item_2_components}
         Item 2 Description: {item_2_description}
         Operation: {operation}""",
            ),
        ]
    )
    llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    parser = PydanticOutputParser(pydantic_object=ItemNameAndDescription)
    format_instructions = parser.get_format_instructions()
    llm_chain = prompt | llm | parser

    item_name_and_description: ItemNameAndDescription = await llm_chain.ainvoke(
        {
            "item_1_name": item_1.name,
            "item_1_components": item_1.components,
            "item_1_description": item_1.description,
            "item_2_name": item_2.name,
            "item_2_components": item_2.components,
            "item_2_description": item_2.description,
            "operation": operation,
            "format_instructions": format_instructions,
        }
    )

    return Item(
        code=combined_code,
        components=combined_components,
        name=item_name_and_description.name,
        description=item_name_and_description.description,
    )


async def generate_description(name: str) -> str:
    """
    Generate a description for the item based on its name.

    :param name: Name of the item
    :return: A generated description string
    """
    item_description_prompt = None
    async with aiofiles.open(
        f"{PROMPTS_DIRECTORY}/alchemy/item_description_generator.md"
    ) as f:
        item_description_prompt = await f.read()

    prompt = ChatPromptTemplate([("system", item_description_prompt), ("user", name)])
    llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    llm_chain = prompt | llm | StrOutputParser()

    return await llm_chain.ainvoke({})

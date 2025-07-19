from alchemy.models import Item, AlchemyCode, Operation, generate_alchemy_code
from alchemy.operations import alchemy_and, alchemy_or
from database.alchemy_database import get_item_by_name_or_code, insert_item
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import aiofiles

async def alchemize_items(item_1_name: str, item_2_name: str, operation: Operation) -> Item:
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
        item_1 = await new_item(item_1_name, None, None)
    if not item_2:
        item_2 = await new_item(item_2_name, None, None)

    name = await generate_item_name(item_1_name, item_2_name, operation)
    print(name)
    description = await generate_description(name, operation)

    combined_code = None
    match operation:
        case 'and':
            combined_code = alchemy_and(item_1.code, item_2.code)
        case 'or':
            combined_code = alchemy_or(item_1.code, item_2.code)
        case _:
            raise ValueError(f"Invalid operation: {operation}. Use 'and' or 'or'.")
        
    combined_item = await new_item(name, description, combined_code)
    await insert_item(combined_item)

    return combined_item

async def new_item(name: str, description: str | None, code: AlchemyCode | None) -> Item:
    """
    Create a new item with a unique alchemy code.
    
    :param name: Name of the item
    :param description: Description of the item (optional)
    :return: An Item instance with a generated alchemy code
    """
    if not description:
        description = await generate_description(name, None)
    if not code:
        code = generate_alchemy_code()
    return Item(name=name, description=description or "", code=code)

async def generate_item_name(name_1: str, name_2: str, operation: Operation) -> str:
    """
    Generate a name for the new item based on the names of the two items being combined.
    
    :param name_1: Name of the first item
    :param name_2: Name of the second item
    :param operation: The operation used ('and' or 'or')
    :return: A generated name string
    """
    llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    item_name_prompt = None
    async with aiofiles.open(f'prompts/alchemy/item_name_generator.md') as f:
        item_name_prompt = await f.read()
    prompt = ChatPromptTemplate([
        ('system', item_name_prompt),
        ('user', f'{name_1}, {name_2}, operation: {operation}')
    ])
    llm_chain = prompt | llm | StrOutputParser()
    return await llm_chain.ainvoke({})

async def generate_description(name: str, operation: Operation | None) -> str:
    """
    Generate a description for the item based on its name.
    
    :param name: Name of the item
    :return: A generated description string
    """
    if not operation:
        operation = 'new item'
    llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")
    item_description_prompt = None
    async with aiofiles.open(f'prompts/alchemy/item_description_generator.md') as f:
        item_description_prompt = await f.read()
    prompt = ChatPromptTemplate([
        ('system', item_description_prompt),
        ('user', f'{name}, operation: {operation}')
    ])
    llm_chain = prompt | llm | StrOutputParser()
    return await llm_chain.ainvoke({})
    
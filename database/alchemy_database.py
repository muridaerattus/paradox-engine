from dotenv import load_dotenv, find_dotenv
from alchemy.models import Item, format_name
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select


load_dotenv(find_dotenv())

DATABASE_LINK = os.environ.get('DATABASE_LINK')
if not DATABASE_LINK:
    raise ValueError("DATABASE_LINK environment variable is not set.")
engine = create_async_engine(DATABASE_LINK, echo=True)
async_session = AsyncSession(engine, expire_on_commit=False)


async def get_item_by_name_or_code(name_or_code: str) -> Item | None:
    """
    Retrieve an item by its name or alchemy code.
    
    :param name_or_code: The name or code of the item to retrieve
    :return: An Item instance if found, otherwise None
    """
    async with async_session as session:
        async with session.begin():
            result = await session.exec(
                select(Item).where(
                    (Item.code == name_or_code)
                )
            )
            item = result.first()
            if result is None:
                # If no item found by code, try by name
                name_or_code = format_name(name_or_code)
                result = await session.exec(
                    select(Item).where(
                        (Item.name == name_or_code)
                    )
                )
            item = result.first()
            return item
        
async def get_item_by_code(code: str) -> Item | None:
    """
    Retrieve an item by its alchemy code or formatted name.
    
    :param code: The code or name of the item to retrieve
    :return: An Item instance if found, otherwise None
    """
    async with async_session as session:
        async with session.begin():
            result = await session.exec(
                select(Item).where(
                    (Item.code == code)
                )
            )
        item = result.first()
        return item
        
        
async def insert_item(item: Item) -> Item:
    """
    Insert a new item into the database.
    
    :param item: An Item instance to insert
    :return: The inserted Item instance
    """
    async with async_session as session:
        async with session.begin():
            session.add(item)
            await session.commit()
            return item
import asyncio
import json
from pathlib import Path

from paradox_engine.alchemy.models import Item
from paradox_engine.alchemy.repository import (
    get_item_by_code,
    insert_item,
    update_item,
)

OBJECTS_FILE = Path(__file__).with_name("example_objects.json")


async def preload_objects():
    """
    Preload example objects into the database.

    This function reads example objects from a JSON file and inserts them into the database
    only if they do not already exist.
    """
    with OBJECTS_FILE.open(encoding="utf-8") as file:
        item_data = json.load(file)["items"]
        for item in item_data:
            existing = await get_item_by_code(item["code"])
            if not existing:
                await insert_item(Item(**item))
            else:
                existing.name = item["name"]
                existing.components = item["components"]
                existing.tagline = item["tagline"]
                existing.description = item["description"]
                await update_item(existing)


if __name__ == "__main__":
    asyncio.run(preload_objects())

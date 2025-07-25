import asyncio
from database.alchemy_database import insert_item, get_item_by_code
from alchemy.models import Item
import json

async def preload_objects():
    """
    Preload example objects into the database.

    This function reads example objects from a JSON file and inserts them into the database
    only if they do not already exist.
    """
    with open('scripts/example_objects.json', 'r') as file:
        item_data = json.load(file)
        item_data = item_data['items']
        for item in item_data:
            
            # Check if item exists by code
            existing = await get_item_by_code(item['code'])
            if not existing:
                new_item = Item(
                    name=item['name'],
                    code=item['code'],
                    components=item['components'],
                    description=item['description']
                )
                await insert_item(new_item)

if __name__ == "__main__":
    asyncio.run(preload_objects())
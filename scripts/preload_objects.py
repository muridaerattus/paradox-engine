from database.alchemy_database import insert_item
from alchemy.models import Item
import json

async def preload_objects():
    """
    Preload example objects into the database.

    This function reads example objects from a JSONL file and inserts them into the database
    only if they do not already exist.
    """
    with open('scripts/example_objects.jsonl', 'r') as file:
        for line in file:
            item_data = json.loads(line)
            # Check if item exists by code
            existing = await Item.get_by_code(item_data['code'])
            if not existing:
                item = Item(
                    name=item_data['name'],
                    code=item_data['code'],
                    components=item_data['components'],
                    description=item_data['description']
                )
                await insert_item(item)
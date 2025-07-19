from pydantic.types import StringConstraints
from typing import Annotated, Literal
import random
import string
from sqlmodel import SQLModel, Field

AlchemyCode = Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9!?]{8}$')]
AlchemyCodeBinary = Annotated[list[int], Field(min_length=8, max_length=8), Field(ge=0, le=63)]
Operation = Literal['and', 'or']

def generate_alchemy_code() -> str:
    chars = string.ascii_letters + string.digits + "!?"
    return ''.join(random.choices(chars, k=8))

def format_name(name: str) -> str:
    """'this is an item name' -> 'This Is An Item Name'"""
    return ' '.join(word.capitalize() for word in name.split())

class Item(SQLModel, table=True):
    id: int = Field(..., description="", primary_key=True)
    code: AlchemyCode = Field(default_factory=generate_alchemy_code, description="8 character alchemy code")
    components: str = Field(default="", description="Chain of components used to create this item")
    name: str = Field(..., description="Name of the item")
    description: str = Field(..., description="Description of the item")

class ItemNameAndDescription(SQLModel):
    thinking_space: str = Field(description="Space to think about the item's name and description")
    name: str = Field(..., description="Final name of the item")
    description: str = Field(..., description="Final description of the item")
from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing import NewType, Annotated, Literal
import random
import string

AlchemyCode = Annotated[str, StringConstraints(pattern=r'^[a-zA-Z0-9!?]{8}$')]
AlchemyCodeBinary = Annotated[list[int], Field(min_length=8, max_length=8), Field(ge=0, le=63)]
Operation = Literal['and', 'or']

def generate_alchemy_code() -> str:
    chars = string.ascii_letters + string.digits + "!?"
    return ''.join(random.choices(chars, k=8))

class Item(BaseModel):
    id: int = Field(..., description="")
    code: AlchemyCode = Field(default_factory=generate_alchemy_code, description="8 character alchemy code")
    name: str = Field(..., description="Name of the item")
    description: str = Field(..., description="Description of the item")
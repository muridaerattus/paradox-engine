from pydantic import BaseModel, Field

class Title(BaseModel):
    title_class: str = Field(..., description="The class of the character.")
    title_aspect: str = Field(..., description="The aspect of the character.")

class Fraymotif(BaseModel):
    thinking_space: str = Field(..., description="Space to think about the fraymotif's themes and mechanics.")
    name: str = Field(..., description="The name of the fraymotif.")
    visual_description: str = Field(..., description="A description of the fraymotif's visual appearance while in use.")
    mechanical_description: str = Field(..., description="A description of what the fraymotif does.")
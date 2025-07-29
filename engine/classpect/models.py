from pydantic import BaseModel, Field


class ParadoxEngineOutput(BaseModel):
    class_result: str = Field(
        description="The class result from the quiz, representing the character's class."
    )
    aspect_result: str = Field(
        description="The aspect result from the quiz, representing the character's aspect."
    )
    llm_response: str = Field(
        description="The response from the language model, containing the final title."
    )

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


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


class ClasspectInterviewDecision(BaseModel):
    response: str = Field(
        description="The next in-character question, or an empty string when complete."
    )
    personality_summary: str = Field(
        description="A complete personality summary when complete, otherwise an empty string."
    )


class ClasspectThread(SQLModel, table=True):
    __tablename__ = "classpect_thread"

    id: str = SQLField(default_factory=lambda: str(uuid4()), primary_key=True)
    status: str = SQLField(default="active", index=True)
    version: int = SQLField(default=0)
    class_result: str | None = None
    aspect_result: str | None = None
    result: str | None = None
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))


class ClasspectThreadMessage(SQLModel, table=True):
    __tablename__ = "classpect_thread_message"

    id: int | None = SQLField(default=None, primary_key=True)
    thread_id: str = SQLField(foreign_key="classpect_thread.id", index=True)
    role: str
    content: str
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))


class ClasspectThreadState(BaseModel):
    thread: ClasspectThread
    messages: list[ClasspectThreadMessage]

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from paradox_engine.alchemy.models import Operation


class StatusResponse(BaseModel):
    message: str


class ClasspectRequest(BaseModel):
    personality: str = Field(min_length=1, max_length=10_000)


class ClasspectResponse(BaseModel):
    class_: str = Field(alias="class")
    aspect: str
    result: str


class ClasspectThreadContinuationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class ClasspectThreadMessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ClasspectThreadResponse(BaseModel):
    thread_id: UUID
    status: Literal["active", "completed"]
    messages: list[ClasspectThreadMessageResponse]
    result: ClasspectResponse | None = None


class AlchemizeRequest(BaseModel):
    item_one: str = Field(min_length=1, max_length=200)
    item_two: str = Field(min_length=1, max_length=200)
    operation: Operation


class ItemResponse(BaseModel):
    name: str
    code: str
    description: str
    tagline: str


class FraymotifRequest(BaseModel):
    players: str = Field(min_length=1, max_length=1_000)
    memory: str = Field(min_length=1, max_length=10_000)
    additional_info: str = Field(default="", max_length=10_000)


class FraymotifResponse(BaseModel):
    visual_description: str
    name: str
    mechanical_description: str

from typing import TypeVar

from openrouter import OpenRouter, components
from pydantic import BaseModel

from prompt_library import ChatMessage
from settings import OPENROUTER_API_KEY


OutputT = TypeVar("OutputT", bound=BaseModel)

client = OpenRouter(api_key=OPENROUTER_API_KEY)


def _response_text(response: components.ChatResult) -> str:
    if not response.choices:
        raise ValueError("OpenRouter returned no choices")

    message = response.choices[0].message
    if isinstance(message.content, str):
        return message.content

    if message.refusal:
        raise ValueError(f"The model refused the request: {message.refusal}")
    raise TypeError("OpenRouter returned a response without text content")


def _strict_json_schema(value: object) -> None:
    """Disallow undeclared properties in every object in a JSON Schema."""
    if isinstance(value, dict):
        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _strict_json_schema(child)
    elif isinstance(value, list):
        for child in value:
            _strict_json_schema(child)


async def generate_text(
    *,
    model: str,
    messages: list[ChatMessage],
    max_tokens: int | None = None,
) -> str:
    if max_tokens is None:
        response = await client.chat.send_async(
            model=model,
            messages=messages,
            stream=False,
        )
    else:
        response = await client.chat.send_async(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
    return _response_text(response)


async def generate_structured(
    *,
    model: str,
    messages: list[ChatMessage],
    output_type: type[OutputT],
    max_tokens: int | None = None,
) -> OutputT:
    schema = output_type.model_json_schema()
    _strict_json_schema(schema)

    provider = components.ProviderPreferences(require_parameters=True)
    response_format = components.ChatFormatJSONSchemaConfig(
        type="json_schema",
        json_schema=components.ChatJSONSchemaConfig(
            name=output_type.__name__,
            schema_=schema,
            strict=True,
        ),
    )

    if max_tokens is None:
        response = await client.chat.send_async(
            model=model,
            messages=messages,
            provider=provider,
            response_format=response_format,
            stream=False,
        )
    else:
        response = await client.chat.send_async(
            model=model,
            messages=messages,
            provider=provider,
            response_format=response_format,
            max_tokens=max_tokens,
            stream=False,
        )
    return output_type.model_validate_json(_response_text(response))

from typing import TypeVar

from openrouter import OpenRouter, components
from pydantic import BaseModel

from paradox_engine.prompts.messages import ChatMessage


OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMClient:
    """Small, testable adapter around the OpenRouter SDK."""

    def __init__(self, api_key: str | None) -> None:
        self.client = OpenRouter(api_key=api_key)

    @staticmethod
    def _provider_preferences() -> components.ProviderPreferences:
        return components.ProviderPreferences(
            require_parameters=True,
            sort="throughput",
        )

    @staticmethod
    def _response_text(response: components.ChatResult) -> str:
        if not response.choices:
            raise ValueError("OpenRouter returned no choices")
        message = response.choices[0].message
        if isinstance(message.content, str):
            return message.content
        if message.refusal:
            raise ValueError(f"The model refused the request: {message.refusal}")
        raise TypeError("OpenRouter returned a response without text content")

    @classmethod
    def _strict_json_schema(cls, value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value.setdefault("additionalProperties", False)
            for child in value.values():
                cls._strict_json_schema(child)
        elif isinstance(value, list):
            for child in value:
                cls._strict_json_schema(child)

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        reasoning_effort: components.ChatRequestReasoningEffort | None = None,
    ) -> str:
        options = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        response = await self.client.chat.send_async(
            model=model,
            messages=messages,
            provider=self._provider_preferences(),
            stream=False,
            **options,
        )
        return self._response_text(response)

    async def generate_structured(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        output_type: type[OutputT],
        max_tokens: int | None = None,
    ) -> OutputT:
        schema = output_type.model_json_schema()
        self._strict_json_schema(schema)
        response_format = components.ChatFormatJSONSchemaConfig(
            type="json_schema",
            json_schema=components.ChatJSONSchemaConfig(
                name=output_type.__name__, schema_=schema, strict=True
            ),
        )
        options = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        response = await self.client.chat.send_async(
            model=model,
            messages=messages,
            provider=self._provider_preferences(),
            response_format=response_format,
            stream=False,
            **options,
        )
        return output_type.model_validate_json(self._response_text(response))

    async def classify_choices(
        self,
        *,
        model: str,
        state: str | dict | list,
        questions: dict[str, dict],
    ) -> dict[str, str]:
        response = await self.client.system_one.create_async(
            model=model, state=state, questions=questions
        )
        choices = {}
        for question_name, question in questions.items():
            answer = response.answers.get(question_name)
            if answer is None:
                raise ValueError(f"Classifier did not answer {question_name}")
            if getattr(answer, "type", None) != "choice":
                raise TypeError(
                    f"Classifier returned a non-choice answer for {question_name}"
                )
            choice = getattr(answer, "choice", None)
            if choice not in question.get("criteria", {}):
                raise ValueError(
                    f"Classifier returned unknown choice {choice!r} for {question_name}"
                )
            choices[question_name] = choice
        return choices

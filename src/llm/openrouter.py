from typing import Type

from langchain_openai import ChatOpenAI

from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    """OpenRouter via the OpenAI-compatible API (ChatOpenAI + custom base_url)."""

    def get_model_class(self) -> Type:
        return ChatOpenAI

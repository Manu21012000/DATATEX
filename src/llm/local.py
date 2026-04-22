from typing import Type

from langchain_openai import ChatOpenAI

from .base import BaseProvider


class LocalOpenAIProvider(BaseProvider):
    """Local LLM via an OpenAI-compatible HTTP API (Ollama, llama.cpp server, LM Studio, etc.)."""

    def get_model_class(self) -> Type:
        return ChatOpenAI

from typing import Any, Type

from langchain_community.chat_models import ChatLlamaCpp

from .base import BaseProvider
from .llamacpp_singleton import get_shared_chat_llamacpp


class LlamaCppProvider(BaseProvider):
    """In-process llama-cpp-python (GGUF) via LangChain ChatLlamaCpp."""

    def get_model_class(self) -> Type:
        return ChatLlamaCpp

    def create_model(self, config: dict[str, Any]) -> Any:
        return get_shared_chat_llamacpp(config)

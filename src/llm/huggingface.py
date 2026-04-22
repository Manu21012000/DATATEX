from typing import Type

from langchain_openai import ChatOpenAI

from .base import BaseProvider


class HuggingFaceProvider(BaseProvider):
    """Hugging Face Inference Providers via the OpenAI-compatible router.

    Uses ``ChatOpenAI`` with ``base_url`` pointing at ``router.huggingface.co/v1``.
    Set ``HF_TOKEN`` in the environment (see ``LanguageModelManager.get_model_config``).
    Model IDs may include a provider suffix, e.g. ``org/model:together``.
    """

    def get_model_class(self) -> Type:
        return ChatOpenAI

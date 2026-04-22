from abc import ABC, abstractmethod
from typing import Any, Type

class BaseProvider(ABC):
    """An abstract base class for LLM providers."""

    @abstractmethod
    def get_model_class(self) -> Type:
        """
        Gets the model class for the provider.

        Returns:
            The class of the language model (e.g., ChatOpenAI).
        """
        pass

    def create_model(self, config: dict[str, Any]) -> Any:
        """Instantiate the chat model. Override for providers that share one loaded model."""
        return self.get_model_class()(**config)
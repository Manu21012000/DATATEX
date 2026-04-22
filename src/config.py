import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

# Project root (repo containing main.py, config/, src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root, then cwd (dotenv does not read ".env Example" — copy that file to ".env")
load_dotenv(_PROJECT_ROOT / ".env", encoding="utf-8")
load_dotenv(Path.cwd() / ".env", encoding="utf-8")
load_dotenv(encoding="utf-8")


def _first_nonempty_env(*names: str):
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


# Set up API keys and environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
# Hugging Face: Inference Providers OpenAI-compatible router (chat completions)
# Accept HF_TOKEN or the standard hub token name used by huggingface_hub / HF docs
HF_TOKEN = _first_nonempty_env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HF_INFERENCE_ROUTER_BASE = os.getenv(
    'HF_INFERENCE_ROUTER_BASE',
    'https://router.huggingface.co/v1',
)
# OpenRouter (OpenAI-compatible): https://openrouter.ai/docs
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv(
    'OPENROUTER_BASE_URL',
    'https://openrouter.ai/api/v1',
)
# Working directory for data files and MCP filesystem roots (absolute path avoids cwd/MCP drift)
_wd_raw = os.getenv("WORKING_DIRECTORY", "data")
_wd_path = Path(_wd_raw)
if _wd_path.is_absolute():
    WORKING_DIRECTORY = str(_wd_path.expanduser().resolve())
else:
    WORKING_DIRECTORY = str((_PROJECT_ROOT / _wd_path).resolve())
# MCP config expands ${WORKING_DIRECTORY} from os.environ — keep it aligned with the resolved path
os.environ["WORKING_DIRECTORY"] = WORKING_DIRECTORY
# Get Conda-related paths from environment variables
CONDA_ENV = os.getenv('CONDA_ENV', 'base')
# Get ChromeDriver
CHROMEDRIVER_PATH = os.getenv('CHROMEDRIVER_PATH', './chromedriver/chromedriver')

# Exposed for startup hints (e.g. missing .env)
PROJECT_ROOT = _PROJECT_ROOT


class AgentModelsConfig:
    """Configuration class for loading agent models from YAML file."""

    def __init__(self, config_path: str = 'config/agent_models.yaml'):
        """Initialize the configuration by loading the YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {config_path}") from e

    @property
    def agents(self):
        """Get the agents configuration."""
        return self._config.get('agents', {})

    def get_agent_config(self, agent_name: str):
        """Get configuration for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Agent configuration dictionary or empty dict if not found.
        """
        return self.agents.get(agent_name, {})

    def get_provider(self, agent_name: str):
        """Get the provider for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Provider name or None if not found.
        """
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('provider')

    def get_model_config(self, agent_name: str):
        """Get the model configuration for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Model configuration dictionary or empty dict if not found.
        """
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get('model_config', {})


# Create global instance
AGENT_MODELS = AgentModelsConfig()
import multiprocessing
import os
from pathlib import Path

from ..logger import setup_logger
from ..llm.factory import ProviderFactory
from ..config import (
    AGENT_MODELS,
    GOOGLE_API_KEY,
    HF_INFERENCE_ROUTER_BASE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROJECT_ROOT,
)


def _hf_token_from_env() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        raw = os.getenv(key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


class LanguageModelManager:
    def __init__(self):
        """Initialize the language model manager"""
        self.logger = setup_logger()
        self.provider_factory = ProviderFactory()

    def get_provider(self, agent_name: str):
        """Get the provider for the given agent."""
        provider_name = AGENT_MODELS.get_provider(agent_name)
        if not provider_name:
            raise ValueError(f"No provider configured for agent '{agent_name}'")
        return self.provider_factory.create_provider(provider_name)

    def get_model_config(self, agent_name: str) -> dict:
        """Get the model configuration for the given agent."""
        config = AGENT_MODELS.get_model_config(agent_name)
        if not config:
            raise ValueError(f"No model config configured for agent '{agent_name}'")
        provider_name = AGENT_MODELS.get_provider(agent_name)
        if provider_name == "huggingface":
            merged = dict(config)
            merged.setdefault("base_url", HF_INFERENCE_ROUTER_BASE)
            # Resolve at call time so shell/env changes after import still work
            api_key = merged.get("api_key") or _hf_token_from_env()
            if not api_key:
                env_path = PROJECT_ROOT / ".env"
                missing_file = not env_path.is_file()
                hint = ""
                if missing_file:
                    hint = (
                        f" Create a file named .env at {env_path} "
                        f'(copy from ".env Example") and set HF_TOKEN=your_token.'
                    )
                raise ValueError(
                    "Hugging Face inference requires a token: set HF_TOKEN or "
                    "HUGGING_FACE_HUB_TOKEN (e.g. in .env), or api_key under that "
                    "agent's model_config in config/agent_models.yaml."
                    + hint
                )
            merged["api_key"] = api_key
            return merged
        if provider_name == "google":
            merged = dict(config)
            api_key = merged.get("google_api_key") or GOOGLE_API_KEY
            if not api_key:
                raise ValueError(
                    "Google Gemini requires GOOGLE_API_KEY in .env (Google AI Studio / "
                    "Gemini API key), or google_api_key under that agent's model_config."
                )
            merged["google_api_key"] = api_key
            return merged
        if provider_name == "openrouter":
            merged = dict(config)
            merged.setdefault("base_url", OPENROUTER_BASE_URL)
            api_key = merged.get("api_key") or OPENROUTER_API_KEY
            if not api_key:
                raise ValueError(
                    "OpenRouter requires OPENROUTER_API_KEY in .env (from "
                    "https://openrouter.ai/keys), or api_key under that agent's "
                    "model_config in config/agent_models.yaml."
                )
            merged["api_key"] = api_key
            # Optional OpenRouter attribution (recommended on their dashboard)
            headers = dict(merged.get("default_headers") or {})
            referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.getenv("OPENROUTER_APP_TITLE", "DATAGEN").strip()
            if referer:
                headers.setdefault("HTTP-Referer", referer)
            if title:
                headers.setdefault("X-Title", title)
            if headers:
                merged["default_headers"] = headers
            return merged
        if provider_name == "local":
            merged = dict(config)
            # llama.cpp llama-server (default): http://127.0.0.1:8080/v1 — see scripts/run_llama_server.*
            # Ollama: set LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1 and LOCAL_LLM_API_KEY=ollama
            merged.setdefault(
                "base_url",
                os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
            )
            merged.setdefault(
                "api_key",
                os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
            )
            return merged
        if provider_name == "llamacpp":
            merged = dict(config)
            raw_path = merged.pop("model_path", None) or os.getenv("LLAMA_CPP_MODEL_PATH")
            if not raw_path:
                raise ValueError(
                    "Provider llamacpp requires model_path in model_config (path to the .gguf "
                    "file, relative to the project root or absolute) or LLAMA_CPP_MODEL_PATH."
                )
            p = Path(raw_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            p = p.resolve()
            if not p.is_file():
                raise ValueError(f"GGUF not found: {p}")
            out: dict = {"model_path": str(p)}
            for key, val in merged.items():
                if val is not None:
                    out[key] = val
            out.setdefault("n_ctx", int(os.getenv("LLAMA_CPP_N_CTX", "8192")))
            out.setdefault("n_gpu_layers", int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "0")))
            if "n_threads" not in out:
                nt = os.getenv("LLAMA_CPP_N_THREADS")
                if nt and str(nt).strip():
                    out["n_threads"] = max(1, int(nt))
                else:
                    out["n_threads"] = max(1, multiprocessing.cpu_count() - 1)
            out.setdefault("temperature", 0.45)
            out.setdefault("max_tokens", 8192)
            out.setdefault("streaming", False)
            out.setdefault("verbose", os.getenv("LLAMA_CPP_VERBOSE", "").lower() in ("1", "true", "yes"))
            return out
        return config

"""Single in-process ChatLlamaCpp instance so the GGUF is loaded once for all agents."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None
_loaded_path: Optional[str] = None


def get_shared_chat_llamacpp(config: dict[str, Any]) -> Any:
    """Return one shared ChatLlamaCpp; first call loads the GGUF."""
    global _model, _loaded_path

    path = str(config.get("model_path", ""))
    with _lock:
        if _model is not None:
            if path and _loaded_path and path != _loaded_path:
                logger.warning(
                    "llamacpp: ignoring different model_path (already loaded %s); "
                    "reuse first instance.",
                    _loaded_path,
                )
            return _model

        from langchain_community.chat_models import ChatLlamaCpp

        logger.info("Loading ChatLlamaCpp (first agent): %s", path)
        _model = ChatLlamaCpp(**config)
        _loaded_path = path
        return _model

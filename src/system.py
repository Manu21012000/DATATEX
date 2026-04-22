import os
from typing import Dict, Any
from . import config, logger
from langchain_core.messages import HumanMessage

from .core import WorkflowManager, LanguageModelManager

class MultiAgentSystem:
    def __init__(self):
        self.logger = logger.setup_logger()
        self.setup_environment()
        self.lm_manager = LanguageModelManager()
        self.workflow_manager = WorkflowManager(
            lm_manager=self.lm_manager,
            working_directory=config.WORKING_DIRECTORY
        )

    def setup_environment(self):
        """Initialize environment variables"""
        if config.OPENAI_API_KEY is not None:
            os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        if config.LANGCHAIN_API_KEY is not None:
            os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
        if config.HF_TOKEN is not None:
            os.environ.setdefault("HF_TOKEN", config.HF_TOKEN)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", config.HF_TOKEN)
        if config.GOOGLE_API_KEY is not None:
            os.environ.setdefault("GOOGLE_API_KEY", config.GOOGLE_API_KEY)
        if config.OPENROUTER_API_KEY is not None:
            os.environ.setdefault("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
        # LangSmith returns 403 if tracing is on with a placeholder key
        lc = os.getenv("LANGCHAIN_API_KEY", "").strip()
        looks_real = bool(lc) and len(lc) >= 20 and "XXXX" not in lc.upper()
        if looks_real:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "Multi-Agent Data Analysis System"
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"

        if not os.path.exists(config.WORKING_DIRECTORY):
            os.makedirs(config.WORKING_DIRECTORY)
            self.logger.info(f"Created working directory: {config.WORKING_DIRECTORY}")

    def run(self, user_input: str) -> None:
        """Run the multi-agent system with user input"""
        graph = self.workflow_manager.get_graph()
        events = graph.stream(
            {
                "messages": [HumanMessage(content=user_input)],
                "hypothesis": "",
                "process_decision": "",
                "process": "",
                "visualization_state": "",
                "searcher_state": "",
                "code_state": "",
                "report_section": "",
                "quality_review": "",
                "needs_revision": False,
                "last_sender": "",
            },
            {"configurable": {"thread_id": "1"}, "recursion_limit": 3000},
            stream_mode="values",
            debug=False
        )
        
        for event in events:
            message = event["messages"][-1]
            if isinstance(message, tuple):
                print(message, end='', flush=True)
            else:
                message.pretty_print()
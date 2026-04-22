from typing import Any
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import logging
import json
import re
from pathlib import Path

from .state import State
from ..config import WORKING_DIRECTORY

# Set up logger
logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Remove common markdown code fences so JSON can be parsed."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _last_ai_message_content(result: dict[str, Any]) -> str:
    """Text from the last AI message in an agent invoke result."""
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) in ("ai", "AIMessage"):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                    else:
                        parts.append(str(block))
                return "".join(parts)
            return str(content)
    return ""


def _fallback_process_output(text: str) -> Any:
    """Build a ProcessRouteSchema-compatible object when structured_response is missing."""
    from ..agents.process_agent import ProcessRouteSchema

    cleaned = _strip_json_fences(text)
    if cleaned:
        try:
            return ProcessRouteSchema.model_validate_json(cleaned)
        except Exception:
            try:
                return ProcessRouteSchema.model_validate(json.loads(cleaned))
            except Exception:
                logger.warning(
                    "Could not parse process_agent output as JSON; using plain-text fallback."
                )
    return SimpleNamespace(task=text or "Continue the analysis.", next="Coder")


def _fallback_quality_output(text: str) -> Any:
    """Build a QualityOutput-compatible object when structured_response is missing."""
    from ..agents.quality_review_agent import QualityOutput

    cleaned = _strip_json_fences(text)
    if cleaned:
        try:
            return QualityOutput.model_validate_json(cleaned)
        except Exception:
            try:
                return QualityOutput.model_validate(json.loads(cleaned))
            except Exception:
                logger.warning(
                    "Could not parse quality_review_agent output as JSON; using plain-text fallback."
                )
    low = (text or "").lower()
    needs = any(
        w in low
        for w in ("revise", "revision", "incorrect", "missing", "improve", "error", "issue")
    )
    return QualityOutput(feedback=text or "(No feedback text.)", needs_revision=needs)


def _fallback_note_output(state: State, result: dict[str, Any], name: str) -> Any:
    """Synthesize note-agent structured output from messages + prior state."""
    text = _last_ai_message_content(result)
    summary_msg = AIMessage(content=text, name=name) if text else AIMessage(content="", name=name)
    return SimpleNamespace(
        messages=[summary_msg],
        hypothesis=str(state.get("hypothesis", "")),
        process=str(state.get("process", "")),
        process_decision=str(state.get("process_decision", "")),
        visualization_state=str(state.get("visualization_state", "")),
        searcher_state=str(state.get("searcher_state", "")),
        code_state=str(state.get("code_state", "")),
        report_section=str(state.get("report_section", "")),
        quality_review=str(state.get("quality_review", "")),
        needs_revision=bool(state.get("needs_revision", False)),
        sender="note_agent",
    )


def agent_node(state: State, agent: Any, name: str) -> dict:
    """Process an agent's action and update the state accordingly.
    
    Args:
        state: The current state of the workflow.
        agent: The agent instance to invoke.
        name: The name of the agent, used for routing and logging.
        
    Returns:
        A dictionary containing state updates, including messages and 
        agent-specific output keys.
    """
    logger.info(f"Processing agent: {name}")
    try:
        result = agent.invoke(state)
        if not isinstance(result, dict):
            result = {"messages": [], "structured_response": None}

        if name == "process_agent":
            output = result.get("structured_response")
            if output is None:
                output = _fallback_process_output(_last_ai_message_content(result))
            ai_message = AIMessage(content=output.task, name=name)
        elif name == "quality_review_agent":
            output = result.get("structured_response")
            if output is None:
                output = _fallback_quality_output(_last_ai_message_content(result))
            ai_message = AIMessage(content=output.feedback, name=name)
        else:
            messages = result.get("messages") or []
            if not messages:
                raise ValueError("Agent returned no messages")
            message = messages[-1]
            output = message.content
            ai_message = AIMessage(content=output, name=name)

        updates = {
            "messages": [ai_message],
            "sender": name
        }

        if name == "hypothesis_agent":
            updates["hypothesis"] = output
        elif name == "process_agent":
            updates["process"] = output.task
            updates["process_decision"] = output.next
        elif name == "visualization_agent":
            updates["visualization_state"] = output
        elif name == "search_agent":
            updates["searcher_state"] = output
        elif name == "report_agent":
            updates["report_section"] = output
        elif name == "quality_review_agent":
            updates["quality_review"] = output.feedback
            updates["needs_revision"] = output.needs_revision
        
        return updates
        
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}", name=name)],
            "sender": name
        }

def human_choice_node(state: State) -> dict:
    """Handle human input to choose the next step.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary containing state updates based on user choice.
    """
    print("Please choose the next step:")
    print("1. Regenerate hypothesis")
    print("2. Continue the research process")
    
    while True:
        choice = input("Please enter your choice (1 or 2): ")
        if choice in ["1", "2"]:
            break
        print("Invalid input, please try again.")
    
    updates = {
        "messages": [],
        "sender": "human"
    }
    
    if choice == "1":
        modification_areas = input("Specify areas to modify: ")
        updates["messages"] = [HumanMessage(content=f"Regenerate hypothesis. Areas: {modification_areas}")]
        updates["hypothesis"] = ""  # Clear hypothesis
    else:
        updates["messages"] = [HumanMessage(content="Continue the research process")]
        updates["process"] = "Continue the research process"
    
    return updates

def create_message(message: BaseMessage, name: str) -> BaseMessage:
    """
    Create a BaseMessage object based on the message type.
    """
    content = message.content
    message_type = message.type.lower()
    
    logger.debug(f"Creating message of type {message_type} for {name}")
    return HumanMessage(content=content) if message_type == "human" else AIMessage(content=content, name=name)

def note_agent_node(state: State, agent: Any, name: str) -> State:
    """Process the note agent's action and update the entire state.
    
    Args:
        state: The current state of the workflow.
        agent: The note agent instance.
        name: The name of the agent.

    Returns:
        The fully updated State object.
    
    Raises:
        Exception: If an unexpected error occurs, returns an error state.
    """
    logger.info(f"Processing note agent: {name}")
    output = ""
    try:
        current_messages = list(state.get("messages", []))
        
        head_messages: list[BaseMessage] = []
        tail_messages: list[BaseMessage] = []
        
        if len(current_messages) > 6:
            head_messages = list(current_messages[:2]) 
            tail_messages = list(current_messages[-2:])
            state = {**state, "messages": list(current_messages[2:-2])}
            logger.debug("Trimmed messages for processing")
        
        result = agent.invoke(state)
        logger.debug(f"Note agent {name} result: {result}")
        if not isinstance(result, dict):
            result = {}
        output = result.get("structured_response")
        if output is None:
            output = _fallback_note_output(state, result, name)

        new_messages = [create_message(msg, name) for msg in output.messages]
        
        messages: list[BaseMessage] = list(new_messages) if new_messages else list(current_messages)
        
        combined_messages = head_messages + messages + tail_messages
        
        updated_state: State = {
            "messages": combined_messages,
            "hypothesis": str(output.hypothesis),
            "process": str(output.process),
            "process_decision": str(output.process_decision),
            "visualization_state": str(output.visualization_state),
            "searcher_state": str(output.searcher_state),
            "code_state": str(output.code_state),
            "report_section": str(output.report_section),
            "quality_review": str(output.quality_review),
            "needs_revision": bool(output.needs_revision),
            "sender": 'note_agent'
        }
        
        logger.info("Updated state successfully")
        return updated_state

    except Exception as e:
        logger.error(f"Unexpected error in note_agent_node: {e}", exc_info=True)
        return _create_error_state(state, AIMessage(content=f"Unexpected error: {str(e)}", name=name), name, "Unexpected error")

def _create_error_state(state: State, error_message: AIMessage, name: str, error_type: str) -> State:
    """
    Create an error state when an exception occurs.
    """
    logger.info(f"Creating error state for {name}: {error_type}")
    error_state:State = {
            "messages": list(state.get("messages", [])) + [error_message],
            "hypothesis": str(state.get("hypothesis", "")),
            "process": str(state.get("process", "")),
            "process_decision": str(state.get("process_decision", "")),
            "visualization_state": str(state.get("visualization_state", "")),
            "searcher_state": str(state.get("searcher_state", "")),
            "code_state": str(state.get("code_state", "")),
            "report_section": str(state.get("report_section", "")),
            "quality_review": str(state.get("quality_review", "")),
            "needs_revision": bool(state.get("needs_revision", False)),
            "sender": 'note_agent'
        }
    return error_state

def human_review_node(state: State) -> dict:
    """Display current state to the user and update the state based on user input.

    Includes error handling for robustness.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary containing state updates representing the user's decision.
    """
    try:
        print("Current research progress:")
        print(state)
        print("\nDo you need additional analysis or modifications?")
        
        while True:
            user_input = input("Enter 'yes' to continue analysis, or 'no' to end the research: ").lower()
            if user_input in ['yes', 'no']:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")
        
        updates: dict[str, Any] = {"sender": "human"}
        
        if user_input == 'yes':
            while True:
                additional_request = input("Please enter your request: ").strip()
                if additional_request:
                    updates["messages"] = [HumanMessage(content=additional_request)]
                    updates["needs_revision"] = True
                    break
                print("Request cannot be empty.")
        else:
            updates["needs_revision"] = False
        
        return updates
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error: {str(e)}", name="human_review")]}

def refiner_node(state: State, agent: Any, name: str) -> dict:
    """Read contents of report materials and process with the refiner agent.

    Read MD file contents and PNG file names from the specified storage path,
    add them as report materials to a new message,
    then process with the agent and update the original state.
    If token limit is exceeded, use only MD file names instead of full content.

    Args:
        state: The current state of the workflow.
        agent: The refiner agent instance.
        name: The name of the agent.

    Returns:
        A dictionary containing the refiner agent's response message.
    """
    try:
        # Get storage path
        storage_path = Path(WORKING_DIRECTORY)
        
        # Collect materials
        materials = []
        md_files = list(storage_path.glob("*.md"))
        png_files = list(storage_path.glob("*.png"))
        
        # Process MD files
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                materials.append(f"MD file '{md_file.name}':\n{f.read()}")
        
        # Process PNG files
        materials.extend(f"PNG file: '{png_file.name}'" for png_file in png_files)
        
        # Combine materials
        combined_materials = "\n\n".join(materials)
        report_content = f"Report materials:\n{combined_materials}"
        
        # Create refiner state
        refiner_state = state.copy()
        refiner_state["messages"] = [HumanMessage(content=report_content)]
        
        result = agent.invoke(refiner_state)
        message = result.get("messages")[-1]
        output = message.content
        ai_message = AIMessage(content=output , name=name)
        
        return {
            "messages": [ai_message],
            "sender": name
        }
        
    except Exception as e:
        logger.error(f"Error in refiner_node: {str(e)}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error: {str(e)}", name=name)]}
    
logger.info("Agent processing module initialized")
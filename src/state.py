from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages:        Annotated[list[AnyMessage], add_messages]
    active_agent:    str
    handoff_count:   int
    user_id:         str
    memory_context:  str


def sanitize_messages(messages: list) -> list:
    """Remove dangling ToolMessages that have no matching AIMessage tool_calls."""
    valid: list = []
    ai_tool_call_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                ai_tool_call_ids.add(tc["id"])
            valid.append(msg)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in ai_tool_call_ids:
                valid.append(msg)
                ai_tool_call_ids.discard(msg.tool_call_id)
        else:
            valid.append(msg)

    return valid


def describe_messages(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"  📚 Memory  : {str(m.content)[:80]}")
        elif isinstance(m, HumanMessage):
            lines.append(f"  👤 Human   : {str(m.content)[:80]}")
        elif isinstance(m, AIMessage):
            tc = [t["name"] for t in (m.tool_calls or [])]
            body = str(m.content)[:60] if m.content else ""
            suffix = f" → calls: {tc}" if tc else ""
            lines.append(f"  🤖 AI      : {body}{suffix}")
        elif isinstance(m, ToolMessage):
            lines.append(f"  🔧 Tool    : [{m.tool_call_id[:12]}…] {str(m.content)[:60]}")
    return "\n".join(lines)

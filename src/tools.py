import time

from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from .config import MAX_HANDOFFS
from .logger import log
from .stats import stats
from .subagents import debug_subagent, explain_subagent, grammar_subagent


# ──────────────────────────────────────────────
#  Loop Guard  (module-level counter — reliable across subgraphs)
# ──────────────────────────────────────────────
_turn_handoff_count = 0


def reset_handoff_count():
    global _turn_handoff_count
    _turn_handoff_count = 0


def _check_loop_guard(target: str) -> str | None:
    """
    Returns a fallback message if the handoff limit is reached,
    or None if the handoff is allowed.
    """
    global _turn_handoff_count
    if _turn_handoff_count >= MAX_HANDOFFS:
        stats.record_blocked()
        msg = (
            f"⚠️ Handoff limit reached ({MAX_HANDOFFS}). "
            f"I'll answer directly instead of transferring to {target}. "
            "Please ask one focused question at a time for best results."
        )
        log("🚫 BLOCKED", "router",
            f"Handoff to {target} blocked (limit={MAX_HANDOFFS})",
            f"handoff_count={_turn_handoff_count}")
        return msg
    _turn_handoff_count += 1
    return None


# ──────────────────────────────────────────────
#  Subagent Tools
# ──────────────────────────────────────────────
@tool("explain_topic", description="Explain a concept or topic clearly with examples")
def call_explain(query: str) -> str:
    """Delegate to the Explain subagent."""
    log("🔧 TOOL", "study_agent", "explain_topic called", f"query: {query[:80]}")
    stats.record_tool("explain_topic")
    t0 = time.time()
    result = explain_subagent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    answer = result["messages"][-1].content
    log("✅ TOOL", "study_agent", f"explain_topic done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


@tool("debug_python_code", description="Debug and fix Python code issues")
def call_debug(query: str) -> str:
    """Delegate to the Debug subagent."""
    log("🔧 TOOL", "coding_agent", "debug_python_code called", f"query: {query[:80]}")
    stats.record_tool("debug_python_code")
    t0 = time.time()
    result = debug_subagent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    answer = result["messages"][-1].content
    log("✅ TOOL", "coding_agent", f"debug_python_code done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


@tool("improve_writing", description="Improve grammar, clarity, and style of text")
def call_grammar(query: str) -> str:
    """Delegate to the Grammar subagent."""
    log("🔧 TOOL", "writing_agent", "improve_writing called", f"query: {query[:80]}")
    stats.record_tool("improve_writing")
    t0 = time.time()
    result = grammar_subagent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    answer = result["messages"][-1].content
    log("✅ TOOL", "writing_agent", f"improve_writing done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


# ──────────────────────────────────────────────
#  Handoff Tools
# ──────────────────────────────────────────────
@tool
def transfer_to_study(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Study Agent for concept explanations and learning help."""
    blocked = _check_loop_guard("study_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → study_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "study_agent")

    return Command(
        goto="study_agent",
        update={
            "active_agent":  "study_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Study Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_coding(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Coding Agent for Python code debugging and fixing."""
    blocked = _check_loop_guard("coding_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → coding_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "coding_agent")

    return Command(
        goto="coding_agent",
        update={
            "active_agent":  "coding_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Coding Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_writing(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Writing Agent for text improvement and grammar fixes."""
    blocked = _check_loop_guard("writing_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → writing_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "writing_agent")

    return Command(
        goto="writing_agent",
        update={
            "active_agent":  "writing_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Writing Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )

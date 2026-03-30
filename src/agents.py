import time

from langchain.agents import create_agent

from .config import MODEL
from .logger import log
from .state import State, describe_messages, sanitize_messages
from .stats import stats
from .tools import (
    call_debug,
    call_explain,
    call_grammar,
    transfer_to_coding,
    transfer_to_study,
    transfer_to_writing,
)


# ──────────────────────────────────────────────
#  Main Agents
# ──────────────────────────────────────────────
study_agent = create_agent(
    model=MODEL,
    name="study_agent",
    tools=[call_explain, transfer_to_coding, transfer_to_writing],
    system_prompt=(
        "You are a Study Agent. Your job is to explain concepts and theory.\n"
        "Always use the explain_topic tool to answer the user.\n\n"
        "Transfer rules — only hand off when the ENTIRE request is out of scope:\n"
        "  • transfer_to_coding  → ONLY if the user wants code written or debugged and there is NO conceptual question to answer.\n"
        "  • transfer_to_writing → ONLY if the user wants text/grammar improved and there is NO concept to explain.\n\n"
        "If the request mixes explaining AND coding (e.g. 'why does this crash AND fix it'), handle the explanation yourself using explain_topic. Do NOT transfer.\n"
        "If both coding and writing are requested, only hand off once."
    ),
)

coding_agent = create_agent(
    model=MODEL,
    name="coding_agent",
    tools=[call_debug, transfer_to_study, transfer_to_writing],
    system_prompt=(
        "You are a Coding Agent. Your job is to debug and fix Python code.\n"
        "Use the debug_python_code tool to analyze and fix code.\n\n"
        "CRITICAL RULES — FOLLOW EXACTLY:\n"
        "1. After calling debug_python_code, compose a FINAL answer and STOP. "
        "NEVER transfer after using your tool.\n"
        "2. If the user says 'explain AND fix' or 'write an answer' → do it ALL yourself. "
        "You can explain code in plain English.\n"
        "3. ONLY transfer if you literally CANNOT help (e.g. user wants grammar check on an essay with zero code).\n"
        "4. When in doubt, answer directly. Do NOT transfer.\n"
    ),
)

writing_agent = create_agent(
    model=MODEL,
    name="writing_agent",
    tools=[call_grammar, transfer_to_study, transfer_to_coding],
    system_prompt=(
        "You are a Writing Agent. Improve grammar, clarity, and style of text.\n\n"
        "CRITICAL RULES — FOLLOW EXACTLY:\n"
        "1. Your output is ALWAYS FINAL. NEVER transfer to another agent.\n"
        "2. NEVER call transfer_to_coding or transfer_to_study. Those tools are disabled for you.\n"
        "3. If the user's message contains code or a code explanation, just improve the writing around it. Do NOT transfer.\n"
        "4. Use improve_writing tool if appropriate, then compose your final answer and STOP.\n"
    ),
)


# ──────────────────────────────────────────────
#  Agent Node Wrappers
# ──────────────────────────────────────────────
def _run_agent(name: str, agent, state: State):
    stats.record_agent(name)
    clean_msgs = sanitize_messages(state["messages"])

    log("▶ START", name, "Running agent",
        f"messages in context: {len(clean_msgs)}  "
        f"handoff_count: {state.get('handoff_count', 0)}\n"
        + describe_messages(clean_msgs[-4:]))

    t0 = time.time()
    result = agent.invoke({**state, "messages": clean_msgs})
    elapsed = time.time() - t0

    out_msgs = result.get("messages", [])
    last = out_msgs[-1] if out_msgs else None
    last_content = (
        (last.content[:150] + "…") if last and last.content
        else "(tool call / no text content)"
    )
    log("■ DONE", name, f"Agent finished ({elapsed:.2f}s)",
        f"output: {last_content}")

    return result


def run_study_agent(state: State):
    return _run_agent("study_agent", study_agent, state)


def run_coding_agent(state: State):
    return _run_agent("coding_agent", coding_agent, state)


def run_writing_agent(state: State):
    return _run_agent("writing_agent", writing_agent, state)

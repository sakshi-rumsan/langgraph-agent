from langchain.agents import create_agent

from .config import MODEL

explain_subagent = create_agent(
    model=MODEL,
    name="explain_subagent",
    system_prompt=(
        "You are an expert tutor. Explain concepts clearly and simply "
        "with examples. Keep responses concise but informative."
    ),
)

debug_subagent = create_agent(
    model=MODEL,
    name="debug_subagent",
    system_prompt=(
        "You are a Python debugging expert. Analyze code, identify bugs, "
        "and provide the corrected code with a brief explanation."
    ),
)

grammar_subagent = create_agent(
    model=MODEL,
    name="grammar_subagent",
    system_prompt=(
        "You are a professional writing editor. Improve grammar, clarity, "
        "and style. Return the improved text with brief notes on changes."
    ),
)

import time
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from .logger import log
from .state import State
from .tools import reset_handoff_count

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def router(
    state: State,
) -> Command[Literal["study_agent", "coding_agent", "writing_agent"]]:
    last_message = state["messages"][-1]

    reset_handoff_count()

    log("▶ START", "router", "Classifying user message",
        f"input: {str(last_message.content)[:100]}")

    t0 = time.time()
    classification = router_llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Classify the user message into exactly one category:\n"
                    "- 'study'   → concept explanation, learning, theory\n"
                    "- 'coding'  → code debugging, fixing Python, programming errors\n"
                    "- 'writing' → text improvement, grammar, formal/informal rewriting\n\n"
                    "If the message mixes categories (e.g. explain AND fix code), "
                    "pick 'coding' as the primary category.\n\n"
                    "Respond with ONLY one word: study, coding, or writing."
                ),
            },
            {"role": "user", "content": last_message.content},
        ]
    )

    category = classification.content.strip().lower()
    target = {
        "study":   "study_agent",
        "coding":  "coding_agent",
        "writing": "writing_agent",
    }.get(category, "study_agent")

    log("■ DONE", "router",
        f"Classified '{category}' → {target} ({time.time()-t0:.2f}s)")

    return Command(
        goto=target,
        update={
            "active_agent":  target,
            "handoff_count": 0,
        },
    )

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
) -> Command[Literal["weather_agent", "places_agent", "itinerary_agent"]]:
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
                    "Classify the user's travel request into exactly one category:\n"
                    "- 'weather'   → Check weather, conditions, rain, temperature, planning advice\n"
                    "- 'places'    → Find attractions, restaurants, landmarks, popular destinations\n"
                    "- 'itinerary' → Create day plan, schedule, activities timing, full itinerary\n\n"
                    "If the message mixes categories (e.g. weather AND places), pick 'itinerary' to create a complete plan.\n\n"
                    "Respond with ONLY one word: weather, places, or itinerary."
                ),
            },
            {"role": "user", "content": last_message.content},
        ]
    )

    category = classification.content.strip().lower()
    target = {
        "weather":   "weather_agent",
        "places":    "places_agent",
        "itinerary": "itinerary_agent",
    }.get(category, "itinerary_agent")

    log("■ DONE", "router",
        f"Classified '{category}' → {target} ({time.time()-t0:.2f}s)")

    return Command(
        goto=target,
        update={
            "active_agent":  target,
            "handoff_count": 0,
        },
    )

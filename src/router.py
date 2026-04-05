import time
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.types import Command

from .logger import log
from .state import State
from .tools import reset_handoff_count

router_llm = ChatOpenAI(model="llama3.1:latest", temperature=0,base_url =  "https://jo3m4y06rnnwhaz.askbhunte.com/v1",api_key='ollama')


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
                    "- 'weather'     → Asking about/checking weather or stating weather preferences\n"
                    "- 'places'      → Asking to find/search attractions, restaurants, landmarks\n"
                    "- 'itinerary'   → Asking to create/plan day schedule with timing\n\n"
                    "Examples:\n"
                    "  'I like hot places' → weather (preference)\n"
                    "  'Find restaurants in Tokyo' → places (request)\n"
                    "  'Create a day plan' → itinerary (request)\n"
                    "  'What's the weather in NYC?' → weather (check)\n\n"
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

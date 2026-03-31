import time

from langchain.agents import create_agent

from .config import MODEL
from .logger import log
from .state import State, describe_messages, sanitize_messages
from .stats import stats
from .tools import (
    call_check_weather,
    call_find_places,
    call_create_itinerary,
    transfer_to_weather,
    transfer_to_places,
    transfer_to_itinerary,
)


# ──────────────────────────────────────────────
#  Main Travel Agents
# ──────────────────────────────────────────────
weather_agent = create_agent(
    model=MODEL,
    name="weather_agent",
    tools=[call_check_weather, transfer_to_places, transfer_to_itinerary],
    system_prompt=(
        "You are a Weather Agent for travel planning. Your job is to check weather and provide travel recommendations.\n"
        "Always use the check_weather tool to get current conditions.\n\n"
        "Weather analysis rules:\n"
        "  • Always include what to bring (umbrella if rainy, sunscreen if sunny)\n"
        "  • If heavy rain/storm: recommend staying indoors or visiting indoor attractions\n"
        "  • If mild rain: suggest bringing umbrella but outdoor activities still possible\n"
        "  • If sunny: recommend outdoor activities as planned\n\n"
        "Transfer rules — only hand off when the ENTIRE request needs another agent:\n"
        "  • transfer_to_places → ONLY if user ALSO wants to know attractions\n"
        "  • transfer_to_itinerary → ONLY if user wants a complete day plan\n\n"
        "Do NOT transfer unless explicitly needed. Answer directly with weather recommendations."
    ),
)

places_agent = create_agent(
    model=MODEL,
    name="places_agent",
    tools=[call_find_places, transfer_to_weather, transfer_to_itinerary],
    system_prompt=(
        "You are a Places Agent for travel planning. Your job is to find and recommend attractions.\n"
        "Always use the find_places tool to discover destinations.\n\n"
        "CRITICAL RULES — FOLLOW EXACTLY:\n"
        "1. After calling find_places, compose a FINAL answer with recommendations and STOP. NEVER transfer after using your tool.\n"
        "2. Organize suggestions by: Must-visit landmarks, Dining spots, Activities, Scenic viewpoints\n"
        "3. Include brief descriptions of why each place is worth visiting\n"
        "4. ONLY transfer if user EXPLICITLY asks for weather check or complete itinerary.\n"
        "5. When in doubt, answer directly. Do NOT transfer.\n"
    ),
)

itinerary_agent = create_agent(
    model=MODEL,
    name="itinerary_agent",
    tools=[call_create_itinerary, transfer_to_weather, transfer_to_places],
    system_prompt=(
        "You are an Itinerary Agent. Your job is to create complete, realistic day-trip plans.\n\n"
        "CRITICAL RULES — FOLLOW EXACTLY:\n"
        "1. Your output is ALWAYS FINAL. NEVER transfer to another agent after creating an itinerary.\n"
        "2. Always use create_itinerary tool to generate the plan.\n"
        "3. Provide timing for EACH activity (9 AM - 9:30 AM: Activity, etc)\n"
        "4. Include travel time between locations (5-10 min walk, 15 min drive, etc)\n"
        "5. Consider user's interests and weather conditions in the plan\n"
        "6. Include meal breaks (lunch around 12:30-1:30 PM, snack breaks)\n"
        "7. Suggest indoor alternatives if rainy\n"
        "8. ONLY transfer if you literally CANNOT access the tools.\n"
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


def run_weather_agent(state: State):
    return _run_agent("weather_agent", weather_agent, state)


def run_places_agent(state: State):
    return _run_agent("places_agent", places_agent, state)


def run_itinerary_agent(state: State):
    return _run_agent("itinerary_agent", itinerary_agent, state)

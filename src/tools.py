import time

from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from .config import MAX_HANDOFFS
from .logger import log
from .stats import stats
from .subagents import weather_subagent, places_subagent, itinerary_subagent
from .weather.weather import get_weather_by_location
from .apify.google_map import fetch_places_from_apify


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
            f"I'll help with your request directly instead of transferring to {target}. "
            "Please ask one focused question at a time for best results."
        )
        log("🚫 BLOCKED", "router",
            f"Handoff to {target} blocked (limit={MAX_HANDOFFS})",
            f"handoff_count={_turn_handoff_count}")
        return msg
    _turn_handoff_count += 1
    return None


# ──────────────────────────────────────────────
#  Subagent Travel Tools
# ──────────────────────────────────────────────
@tool("check_weather", description="Check weather conditions for a travel destination")
def call_check_weather(location: str) -> str:
    """Get weather information for a destination."""
    log("🔧 TOOL", "weather_agent", "check_weather called", f"location: {location[:80]}")
    stats.record_tool("check_weather")
    t0 = time.time()
    
    try:
        weather_data = get_weather_by_location(location)
        if "error" in weather_data:
            result_text = f"Could not find weather data for {location}. Please try another location."
        else:
            result_text = f"Weather for {weather_data['location']['name']}, {weather_data['location']['country']}: " \
                         f"{weather_data['weather']['temperature']}°C, {weather_data['weather']['weather']}. " \
                         f"Humidity: {weather_data['weather']['humidity']}%, Wind: {weather_data['weather']['wind_speed']} m/s"
        
        result = weather_subagent.invoke(
            {"messages": [{"role": "user", "content": f"Analyze this weather for travel planning: {result_text}"}]}
        )
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"Weather API error: {str(e)}. Please try again."
    
    log("✅ TOOL", "weather_agent", f"check_weather done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


@tool("find_places", description="Find popular places and attractions at a destination")
def call_find_places(location: str) -> str:
    """Find attractions and popular places for a destination."""
    log("🔧 TOOL", "places_agent", "find_places called", f"location: {location[:80]}")
    stats.record_tool("find_places")
    t0 = time.time()
    
    try:
        # Fetch places from Apify
        places = fetch_places_from_apify(
            location, 
            ["attractions", "restaurants", "landmarks"],
            max_places=15
        )
        
        if not places:
            places_text = f"No places found for {location}."
        else:
            places_text = f"Found {len(places)} places in {location}:\n"
            for i, place in enumerate(places[:10], 1):
                name = place.get('title', 'Unknown')
                address = place.get('address', 'N/A')
                rating = place.get('review', {}).get('rating', 'N/A') if isinstance(place.get('review'), dict) else place.get('review', 'N/A')
                places_text += f"{i}. {name} ({rating}★) - {address}\n"
        
        result = places_subagent.invoke(
            {"messages": [{"role": "user", "content": f"Recommend the best places to visit: {places_text}"}]}
        )
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"Places API error: {str(e)}. Here are some common attractions: landmarks, monuments, museums, restaurants, cafes, parks, scenic viewpoints."
    
    log("✅ TOOL", "places_agent", f"find_places done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


@tool("create_itinerary", description="Create a complete day trip itinerary with timing")
def call_create_itinerary(location: str, weather: str = "", interests: str = "") -> str:
    """Create a full day itinerary for a destination."""
    log("🔧 TOOL", "itinerary_agent", "create_itinerary called", f"location: {location[:80]}")
    stats.record_tool("create_itinerary")
    t0 = time.time()
    
    try:
        query = f"Create a detailed day itinerary for {location}. "
        if weather:
            query += f"Weather conditions: {weather}. "
        if interests:
            query += f"User interests: {interests}. "
        query += "Structure the day from 9 AM to 5 PM with specific timings, activities, and breaks."
        
        result = itinerary_subagent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"Error creating itinerary: {str(e)}. Please try again."
    
    log("✅ TOOL", "itinerary_agent", f"create_itinerary done ({time.time()-t0:.2f}s)",
        f"response: {answer[:120]}…")
    return answer


# ──────────────────────────────────────────────
#  Handoff Tools
# ──────────────────────────────────────────────
@tool
def transfer_to_weather(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Weather Agent to check weather conditions."""
    blocked = _check_loop_guard("weather_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → weather_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "weather_agent")

    return Command(
        goto="weather_agent",
        update={
            "active_agent":  "weather_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Weather Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_places(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Places Agent to find attractions and destinations."""
    blocked = _check_loop_guard("places_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → places_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "places_agent")

    return Command(
        goto="places_agent",
        update={
            "active_agent":  "places_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Places Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_itinerary(runtime: ToolRuntime) -> Command | str:
    """Transfer to the Itinerary Agent to create a day plan."""
    blocked = _check_loop_guard("itinerary_agent")
    if blocked:
        return blocked

    frm = list(stats.agent_calls.keys())[-1] if stats.agent_calls else "unknown"
    log("🔁 HANDOFF", "router", "Transferring → itinerary_agent",
        f"from: {frm}  handoff_count: {_turn_handoff_count}")
    stats.record_handoff(frm, "itinerary_agent")

    return Command(
        goto="itinerary_agent",
        update={
            "active_agent":  "itinerary_agent",
            "handoff_count": _turn_handoff_count,
            "messages": [
                ToolMessage(
                    content="Transferred to Itinerary Agent",
                    tool_call_id=runtime.tool_call_id,
                ),
            ],
        },
        graph=Command.PARENT,
    )


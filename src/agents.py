import time

from langchain.agents import create_agent

from src.memory_config import add_episodic_memory, get_memory, search_episodic_memory

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


from langchain_core.messages import HumanMessage, AIMessage

memory = get_memory()

# ──────────────────────────────────────────────
#  Main Travel Agents
# ──────────────────────────────────────────────
weather_agent = create_agent(
    model=MODEL,
    name="weather_agent",
    tools=[call_check_weather, transfer_to_places, transfer_to_itinerary],
    system_prompt=(
        "You are a Weather Agent for travel planning. Your job is to understand weather preferences and check weather for destinations.\n\n"
        "PREFERENCE HANDLING:\n"
        "  If user expresses preferences (e.g., 'I like hot/cold places', 'I prefer tropical weather'):\n"
        "    1. Acknowledge their preference explicitly\n"
        "    2. Ask which specific destination they're considering\n"
        "    3. Then use check_weather to verify current conditions match their preference\n"
        "  Use memory context to understand historical preferences and tailor suggestions.\n\n"
        "WEATHER ANALYSIS RULES:\n"
        "  • Always use check_weather tool when a location is mentioned\n"
        "  • Include what to bring (umbrella if rainy, sunscreen if sunny)\n"
        "  • If heavy rain/storm: recommend indoor attractions or rescheduling\n"
        "  • If mild rain: suggest bringing umbrella but outdoor activities still possible\n"
        "  • If sunny/hot: recommend outdoor activities and hydration\n"
        "  • Match recommendations to user's stated weather preferences\n\n"
        "TRANSFER RULES:\n"
        "  • transfer_to_places → ONLY if user wants attraction recommendations\n"
        "  • transfer_to_itinerary → ONLY if user wants a complete day schedule\n\n"
        "Example:\n"
        "  User: 'I like hot places'\n"
        "  You: 'Great! Hot destinations are wonderful. Where are you thinking of visiting? Once you mention a location, I'll check the current weather to ensure it matches your preference.'"
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


# Uncomment to add episodic memory
def run_search_memory_agent(state: State, user_id: str):
    """Search episodic memory for user context based on their query."""
    # Extract the user's query from the latest HumanMessage
    from langchain_core.messages import HumanMessage
    
    query = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    if not query:
        return "No query found in conversation history."
    
    # Search memory for relevant context
    try:
     results = search_episodic_memory(memory, query, user_id=user_id)
    except Exception as e:
        log("❌ MEMORY ERROR", "memory_agent", "Error searching memory", str(e))
        results = ["no memory"]
    
    # Return results or default message
    if results and len(results) > 0:
        return f"Found relevant memory: {results}"
    else:
        return "No memory of user found for this query."


def run_save_memory_agent(state: State, user_id: str):
    """Save the current conversation to episodic memory."""
    from langchain_core.messages import HumanMessage, AIMessage
    
    clean_msgs = sanitize_messages(state["messages"])
    query = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
           
            break
    messages_for_memory = {"role": "user", "content":  query}
    
    # # Convert LangChain messages to dict format that mem0 expects
    # messages_for_memory = []
    # for msg in clean_msgs:
    #     if isinstance(msg, HumanMessage):
    #         messages_for_memory.append({"role": "user", "content": str(msg.content)})
    #     elif isinstance(msg, AIMessage):
    #         messages_for_memory.append({"role": "assistant", "content": str(msg.content)})
    
    if messages_for_memory:
        add_episodic_memory(
            memory, 
            messages_for_memory , 
            user_id=user_id, 
            metadata={"category": "Travel"}
        )
        log("💾 SAVED", "memory_agent", "Conversation saved to episodic memory")
    
    return "Memory saved successfully."
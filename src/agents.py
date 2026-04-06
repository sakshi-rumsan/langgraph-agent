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
from langchain_openai import ChatOpenAI
import os

memory = get_memory()

# Supervisor LLM (same as router)
supervisor_llm = ChatOpenAI(
    model="llama3.1:latest",
    temperature=0,
    base_url=os.getenv("OLLAMA_BASE_URL") + "/v1",
    api_key="ollama",
)

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


# ──────────────────────────────────────────────
#  Supervisor Agent
# ──────────────────────────────────────────────
def run_supervisor(state: State):
    """
    Supervisor agent: searches mem0 for context, classifies the query,
    and either responds directly or routes to travel agents.
    Returns updated state with classification and memory_context.
    """
    user_id = state.get("user_id", "default_user")
    
    # 1. Extract user query
    query = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
    
    if not query:
        return {
            "memory_context": "",
            "active_agent": "supervisor",
        }
    
    # 2. Search mem0 for memory context
    memory_context = ""
    try:
        results = search_episodic_memory(memory, query, user_id=user_id)
        if results and results.get("results"):
            memories = [r.get("memory", "") for r in results["results"]]
            memory_context = "; ".join(memories)
            log("💾 SEARCH", "supervisor", "Retrieved user memory", memory_context[:100])
    except Exception as e:
        log("❌ MEMORY ERROR", "supervisor", "Error searching memory", str(e))
    
    # 3. Classify: travel-tool query or general?
    classification = supervisor_llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a travel assistant supervisor. Classify the user's message:\n"
                "- 'travel' → if they want weather info, place recommendations, or itinerary planning\n"
                "- 'general' → for greetings, chitchat, general questions, preferences, or anything not requiring travel tools\n\n"
                "Examples:\n"
                "  'What is the weather in Paris?' → travel\n"
                "  'Find restaurants in Tokyo' → travel\n"
                "  'Plan my day in London' → travel\n"
                "  'Hello' → general\n"
                "  'What do you remember about me?' → general\n"
                "  'I like spicy food' → general\n"
                "  'Thank you!' → general\n"
                "  'Who are you?' → general\n\n"
                "Respond with ONLY one word: travel or general."
            ),
        },
        {"role": "user", "content": query},
    ])
    
    category = classification.content.strip().lower()
    log("🧠 SUPERVISOR", "supervisor", f"Classified as '{category}'", f"query: {query[:80]}")
    
    # 4. If general → respond directly with memory context
    if category != "travel":
        memory_prompt = ""
        if memory_context:
            memory_prompt = f"\n\nUser memory context: {memory_context}\nUse this to personalize your response."
        
        response = supervisor_llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a friendly travel planning assistant. "
                    "Answer the user's message naturally using any memory context provided. "
                    "Keep responses concise and helpful. "
                    "If the user shares preferences, acknowledge them warmly."
                    f"{memory_prompt}"
                ),
            },
            {"role": "user", "content": query},
        ])
        
        # Save to memory
        try:
            add_episodic_memory(
                memory,
                {"role": "user", "content": query},
                user_id=user_id,
                metadata={"category": "general"}
            )
        except Exception as e:
            log("❌ MEMORY ERROR", "supervisor", "Error saving memory", str(e))
        
        log("🧠 SUPERVISOR", "supervisor", "Responded directly", response.content[:100])
        
        return {
            "messages": [AIMessage(content=response.content)],
            "memory_context": memory_context,
            "active_agent": "supervisor",
        }
    
    # 5. If travel → pass memory context along for router → agents
    return {
        "memory_context": memory_context,
        "active_agent": "router",
    }
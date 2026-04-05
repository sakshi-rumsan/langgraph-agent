from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import SystemMessage

from .agents import (
    run_weather_agent, 
    run_places_agent, 
    run_itinerary_agent,
    run_search_memory_agent,
    run_save_memory_agent
)
from .router import router
from .state import State
from .logger import log

# Wrapper nodes that pass user_id
def search_memory_node(state: State):
    """Search episodic memory for user context."""
    user_id = state.get("user_id", "default_user")
    result = run_search_memory_agent(state, user_id=user_id)
    log("💾 SEARCH", "memory_agent", "Retrieved user memory", result[:100])
    return {"memory_context": result}

def save_memory_node(state: State):
    """Save conversation to episodic memory."""
    user_id = state.get("user_id", "default_user")
    result = run_save_memory_agent(state, user_id=user_id)
    return {"messages": state["messages"]}

def inject_memory_context(state: State, agent_fn):
    """Inject memory context into agent's messages before processing."""
    memory_context = state.get("memory_context", "")
    
    # If memory exists, add it as system context
    if memory_context and memory_context != "No memory of user found for this query.":
        memory_msg = SystemMessage(content=f"📚 User Context from Memory: {memory_context}\n\nUse this context to personalize your recommendations.")
        messages_with_context = [memory_msg] + state["messages"]
        state_with_context = {**state, "messages": messages_with_context}
        return agent_fn(state_with_context)
    
    return agent_fn(state)

def weather_agent_with_memory(state: State):
    """Weather agent with memory context."""
    return inject_memory_context(state, run_weather_agent)

def places_agent_with_memory(state: State):
    """Places agent with memory context."""
    return inject_memory_context(state, run_places_agent)

def itinerary_agent_with_memory(state: State):
    """Itinerary agent with memory context."""
    return inject_memory_context(state, run_itinerary_agent)

builder = StateGraph(State)

builder.add_node("search_memory",    search_memory_node)
builder.add_node("router",           router)
builder.add_node("weather_agent",    weather_agent_with_memory)
builder.add_node("places_agent",     places_agent_with_memory)
builder.add_node("itinerary_agent",  itinerary_agent_with_memory)
builder.add_node("save_memory",      save_memory_node)

builder.add_edge(START, "search_memory")
builder.add_edge("search_memory", "router")
builder.add_edge("weather_agent",    "save_memory")
builder.add_edge("places_agent",     "save_memory")
builder.add_edge("itinerary_agent",  "save_memory")
builder.add_edge("save_memory",      END)

memory = MemorySaver()
graph  = builder.compile(checkpointer=memory)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import SystemMessage

from .agents import (
    run_weather_agent, 
    run_places_agent, 
    run_itinerary_agent,
    run_save_memory_agent,
    run_supervisor,
)
from .router import router
from .state import State
from .logger import log


def supervisor_node(state: State):
    """Supervisor: searches memory, classifies, and either responds or routes."""
    return run_supervisor(state)


def should_route_or_end(state: State):
    """Conditional edge: if supervisor handled it, end. Otherwise, go to router."""
    if state.get("active_agent") == "supervisor":
        return "end"
    return "router"


def save_memory_node(state: State):
    """Save conversation to episodic memory."""
    user_id = state.get("user_id", "default_user")
    result = run_save_memory_agent(state, user_id=user_id)
    return {"messages": state["messages"]}


def inject_memory_context(state: State, agent_fn):
    """Inject memory context into agent's messages before processing."""
    memory_context = state.get("memory_context", "")
    
    if memory_context and memory_context != "No memory of user found for this query.":
        memory_msg = SystemMessage(content=f"📚 User Context from Memory: {memory_context}\n\nUse this context to personalize your recommendations.")
        messages_with_context = [memory_msg] + state["messages"]
        state_with_context = {**state, "messages": messages_with_context}
        return agent_fn(state_with_context)
    
    return agent_fn(state)


def weather_agent_with_memory(state: State):
    return inject_memory_context(state, run_weather_agent)

def places_agent_with_memory(state: State):
    return inject_memory_context(state, run_places_agent)

def itinerary_agent_with_memory(state: State):
    return inject_memory_context(state, run_itinerary_agent)


builder = StateGraph(State)

# Nodes
builder.add_node("supervisor",       supervisor_node)
builder.add_node("router",           router)
builder.add_node("weather_agent",    weather_agent_with_memory)
builder.add_node("places_agent",     places_agent_with_memory)
builder.add_node("itinerary_agent",  itinerary_agent_with_memory)
builder.add_node("save_memory",      save_memory_node)

# Edges: START → supervisor → (end | router → agents → save_memory → END)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    should_route_or_end,
    {"end": END, "router": "router"},
)
builder.add_edge("weather_agent",    "save_memory")
builder.add_edge("places_agent",     "save_memory")
builder.add_edge("itinerary_agent",  "save_memory")
builder.add_edge("save_memory",      END)

memory = MemorySaver()
graph  = builder.compile(checkpointer=memory)

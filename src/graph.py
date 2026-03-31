from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import run_weather_agent, run_places_agent, run_itinerary_agent
from .router import router
from .state import State

builder = StateGraph(State)

builder.add_node("router",           router)
builder.add_node("weather_agent",    run_weather_agent)
builder.add_node("places_agent",     run_places_agent)
builder.add_node("itinerary_agent",  run_itinerary_agent)

builder.add_edge(START, "router")
builder.add_edge("weather_agent",    END)
builder.add_edge("places_agent",     END)
builder.add_edge("itinerary_agent",  END)

memory = MemorySaver()
graph  = builder.compile(checkpointer=memory)

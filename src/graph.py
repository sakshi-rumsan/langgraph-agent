from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import run_coding_agent, run_study_agent, run_writing_agent
from .router import router
from .state import State

builder = StateGraph(State)

builder.add_node("router",        router)
builder.add_node("study_agent",   run_study_agent)
builder.add_node("coding_agent",  run_coding_agent)
builder.add_node("writing_agent", run_writing_agent)

builder.add_edge(START, "router")
builder.add_edge("study_agent",   END)
builder.add_edge("coding_agent",  END)
builder.add_edge("writing_agent", END)

memory = MemorySaver()
graph  = builder.compile(checkpointer=memory)

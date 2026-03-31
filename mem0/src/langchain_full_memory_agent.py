import os
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent  # FIX 1: updated import
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from episodic_memory import add_episodic_memory, search_episodic_memory
from memory_config import get_memory

SKILL_PATH = os.path.join(os.path.dirname(__file__), '../agent-skills/episodic-memory-skill/SKILL.md')

def load_skill_md():
    with open(SKILL_PATH, 'r') as f:
        return f.read()

# ---------------- Tools ----------------

@tool("show_memory_skill_instructions", description="Display instructions from SKILL.md.")
def show_skill_instructions() -> str:
    return load_skill_md()

@tool("add_episodic_memory", description="Store messages in episodic memory. Input: dict with 'messages', 'user_id', optional 'metadata'.")
def episodic_add_tool(input_data: dict) -> str:
    if not input_data or not isinstance(input_data, dict):
        return "Error: input_data is required and must be a dict."
    if 'messages' not in input_data or 'user_id' not in input_data:
        return "Error: 'messages' and 'user_id' fields are required in input_data."
    
    memory = get_memory()
    formatted_messages = [
        {"role": "user", "content": m} if isinstance(m, str) else m
        for m in input_data['messages']
    ]
    add_episodic_memory(memory, formatted_messages, input_data['user_id'], input_data.get('metadata'))
    return "Memory stored successfully."

@tool("search_episodic_memory", description="Retrieve information from episodic memory. Input: dict with 'query' and 'user_id'.")
def episodic_search_tool(input_data: dict) -> str:
    if not input_data or not isinstance(input_data, dict):
        return "Error: input_data is required and must be a dict."
    if 'query' not in input_data or 'user_id' not in input_data:
        return "Error: 'query' and 'user_id' fields are required in input_data."
    
    memory = get_memory()
    results = search_episodic_memory(memory, input_data['query'], input_data['user_id'])
    if results and results.get('results'):
        return str(results['results'])
    return "No memory found."

tools = [
    show_skill_instructions,
    episodic_add_tool,
    episodic_search_tool
]

# ---------------- LLM ----------------

OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = ""

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=float(os.environ.get("OLLAMA_TEMPERATURE", 0)),
    base_url=OLLAMA_URL
)

# ---------------- Agent ----------------

agent = create_react_agent(
    tools=tools,
    model=llm,
    prompt=f"""
You are an intelligent agent with access to episodic memory tools.

Skill Instructions:
{load_skill_md()}

Guidelines:
- Use 'Search Episodic Memory' when you need past information
- Use 'Add Episodic Memory' to store any new preferences, facts, or events
- Decide autonomously when to call these tools
- Respond naturally to the user, but invoke tools when memory access is required
""",
)

# ---------------- Run Loop ----------------

if __name__ == "__main__":
    user_id = os.environ.get("USER_ID", "user123")
    print("=== Autonomous Episodic Memory Agent (Ollama) ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        # FIX 2: wrap under "input_data" key to match the tool's parameter name
        episodic_add_tool.invoke({
            "input_data": {
                "messages": [{"role": "user", "content": query}],
                "user_id": user_id
            }
        })

        response = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        if response.get("messages"):
            last_msg = response["messages"][-1]
            print("AI:", last_msg.content)
        else:
            print("AI: (no response)")
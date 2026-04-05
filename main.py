"""
🌴 Travel Planner Assistant — Multi-Agent System using LangGraph

Architecture:
  User → Router → Travel Agent → Subagent → Response
                   ↕ Handoff (if user needs different travel info)
"""

import time

from langchain_core.messages import AIMessage

from src.config import MAX_TURNS
from src.graph import graph
from src.logger import C, divider, log
from src.stats import stats
from src.evaluator import run_eval


def main():
    divider("═")
    print(f"{C.BOLD}  🌴  Travel Planner Assistant{C.RESET}")
    divider("═")
    print("  Plan your perfect day trip! I can help with:")
    print(f"  {C.STUDY}🌤️  Weather{C.RESET}   → Check weather conditions")
    print(f"  {C.CODING}📍 Places{C.RESET}  → Find popular attractions & destinations")
    print(f"  {C.WRITING}📋 Itinerary{C.RESET} → Create a full day plan")
  
    print("  Type 'quit' to exit  |  'stats' for session summary  |  'eval' to evaluate\n")
    divider()

    # Generate or use user identifier
    user_id = input(f"\n{C.BOLD}Enter your user ID (or press Enter for default):{C.RESET} ").strip() or "default_user"
    config = {"configurable": {"thread_id": f"travel-session-{user_id}"}}
    initial_state = {"user_id": user_id, "memory_context": ""}

    while True:
        try:
            user_input = input(f"\n{C.BOLD}You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            stats.print_summary()
            print("Have a great trip! 🌍")
            break
        if user_input.lower() == "stats":
            stats.print_summary()
            continue
        if user_input.lower() == "eval":
            run_eval()
            continue

        stats.turn += 1
        divider()
        log("💬 USER", "router", f"Turn {stats.turn}",
            f"message: {user_input[:120]}")
        divider()

        t0 = time.time()
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_id": user_id,
            "memory_context": ""
        }
        for turn in range(MAX_TURNS):
            result = graph.invoke(state, config=config)
            messages = result["messages"]
            ai_message = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    ai_message = msg
                    break
            if ai_message:
                total = time.time() - t0
                divider()
                print(f"\n{C.BOLD}Assistant:{C.RESET} {ai_message.content}\n")
                print(f"{C.TIME}  ⏱  total turn time: {total:.2f}s{C.RESET}")
                divider()
                break
            state = result
        else:
            divider()
            print(f"\n{C.BOLD}Assistant:{C.RESET} [No assistant response found after {MAX_TURNS} turns. Possible infinite handoff loop.]")
            divider()


if __name__ == "__main__":
    main()
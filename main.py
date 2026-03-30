"""
Smart Student Assistant — Multi-Agent System using LangGraph

Architecture:
  User → Router → Main Agent → Subagent → Response
                   ↕ Handoff (if user switches topic)
"""

import time

from langchain_core.messages import AIMessage

from src.config import MAX_TURNS
from src.graph import graph
from src.logger import C, divider, log, log_filename
from src.stats import stats
from src.evaluator import run_eval


def main():
    divider("═")
    print(f"{C.BOLD}  🎓  Smart Student Assistant{C.RESET}")
    divider("═")
    print("  I can help you with:")
    print(f"  {C.STUDY}📚 Study{C.RESET}   → Explain concepts")
    print(f"  {C.CODING}💻 Coding{C.RESET}  → Debug Python code")
    print(f"  {C.WRITING}✍️  Writing{C.RESET} → Improve your text")
    print(f"\n  Trace log → {C.SUCCESS}{log_filename}{C.RESET}")
    print("  Type 'quit' to exit  |  'stats' for session summary  |  'eval' to evaluate\n")
    divider()

    config = {"configurable": {"thread_id": "student-session-1"}}

    while True:
        try:
            user_input = input(f"\n{C.BOLD}You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            stats.print_summary()
            print("Goodbye! 👋")
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
        state = {"messages": [{"role": "user", "content": user_input}]}
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
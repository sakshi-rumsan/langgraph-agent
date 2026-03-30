"""
Built-in Graph Evaluation Module

Runs test cases through the graph and scores the outputs using
an LLM judge — no external evaluation platform required.
"""

import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from .config import MAX_TURNS, MODEL
from .graph import graph
from .logger import C, divider


# ──────────────────────────────────────────────
#  Test Cases (built-in dataset)
# ──────────────────────────────────────────────
TEST_CASES: list[dict] = [
    # Study
    {
        "question": "Explain what recursion is in programming.",
        "expected": "Recursion is when a function calls itself to solve smaller instances of the same problem.",
        "domain": "study",
    },
    {
        "question": "What is the difference between a list and a tuple in Python?",
        "expected": "Lists are mutable and use square brackets; tuples are immutable and use parentheses.",
        "domain": "study",
    },
    # Coding
    {
        "question": "Fix this code: def add(a, b) return a + b",
        "expected": "The function is missing a colon after the parameter list: def add(a, b): return a + b",
        "domain": "coding",
    },
    {
        "question": "Why does this crash? print(int('hello'))",
        "expected": "int('hello') raises a ValueError because 'hello' is not a valid integer string.",
        "domain": "coding",
    },
    # Writing
    {
        "question": "Improve this sentence: Me and him went to the store yesterday for buy foods.",
        "expected": "He and I went to the store yesterday to buy food.",
        "domain": "writing",
    },
]


# ──────────────────────────────────────────────
#  Evaluators
# ──────────────────────────────────────────────
_judge_llm = None


def _get_judge():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = init_chat_model(MODEL)
    return _judge_llm


def _extract_answer(messages: list) -> str:
    """Pull the last AI text content from the message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _check_correctness(actual: str, expected: str) -> bool:
    """LLM-as-judge: does the actual answer contain the expected information?"""
    judge = _get_judge()
    response = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Given an actual answer and an expected answer, determine whether "
                    "the actual answer contains all of the information in the expected answer. "
                    "Respond with ONLY the word 'CORRECT' or 'INCORRECT'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ACTUAL ANSWER: {actual}\n\n"
                    f"EXPECTED ANSWER: {expected}"
                ),
            },
        ]
    )
    return response.content.strip().upper() == "CORRECT"


def _check_has_tool_calls(messages: list) -> bool:
    """Did the agent invoke at least one tool?"""
    return any(
        isinstance(msg, AIMessage) and msg.tool_calls
        for msg in messages
    )


def _check_routed(result: dict) -> str:
    """Return which agent was activated, or '?' if unknown."""
    return result.get("active_agent", "?")


# ──────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────
def run_eval(test_cases: list[dict] | None = None):
    """
    Run all test cases through the graph and print a results table.
    Called from the interactive loop via the 'eval' command.
    """
    cases = test_cases or TEST_CASES
    config = {"configurable": {"thread_id": "eval-session"}}

    divider("═")
    print(f"{C.BOLD}  🧪  EVALUATION  ({len(cases)} test cases){C.RESET}")
    divider("═")

    results: list[dict] = []

    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = case["expected"]
        domain = case.get("domain", "?")

        print(f"\n  [{i}/{len(cases)}] {C.DIM}{domain.upper():<8}{C.RESET} {question[:70]}")

        t0 = time.time()
        state = {"messages": [{"role": "user", "content": question}]}

        # Run through the graph (same logic as main loop)
        final_result = state
        for _ in range(MAX_TURNS):
            final_result = graph.invoke(state, config=config)
            messages = final_result.get("messages", [])
            if any(isinstance(m, AIMessage) and m.content for m in reversed(messages)):
                break
            state = final_result

        elapsed = time.time() - t0
        messages = final_result.get("messages", [])
        actual = _extract_answer(messages)
        routed_to = _check_routed(final_result)
        has_tools = _check_has_tool_calls(messages)
        correct = _check_correctness(actual, expected) if actual else False
        not_empty = bool(actual)

        status = f"{C.SUCCESS}PASS{C.RESET}" if correct else f"{C.HANDOFF}FAIL{C.RESET}"
        print(f"           {status}  routed→{routed_to:<14}  tools={has_tools}  "
              f"non-empty={not_empty}  {C.TIME}{elapsed:.2f}s{C.RESET}")
        if not correct:
            print(f"           {C.DIM}expected: {expected[:80]}{C.RESET}")
            print(f"           {C.DIM}actual:   {actual[:80]}{C.RESET}")

        results.append({
            "question": question,
            "domain": domain,
            "correct": correct,
            "not_empty": not_empty,
            "has_tools": has_tools,
            "routed_to": routed_to,
            "elapsed": elapsed,
        })

    # ── Summary ──
    total = len(results)
    passed = sum(1 for r in results if r["correct"])
    failed = total - passed
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0

    divider("─")
    print(f"\n  {C.BOLD}Results:{C.RESET}  {C.SUCCESS}{passed} passed{C.RESET}  "
          f"{C.HANDOFF}{failed} failed{C.RESET}  out of {total}")
    print(f"  {C.BOLD}Accuracy:{C.RESET} {passed/total*100:.0f}%")
    print(f"  {C.BOLD}Avg time:{C.RESET} {avg_time:.2f}s per case")

    # Per-domain breakdown
    domains = sorted(set(r["domain"] for r in results))
    if len(domains) > 1:
        print(f"\n  {C.BOLD}By domain:{C.RESET}")
        for d in domains:
            d_results = [r for r in results if r["domain"] == d]
            d_passed = sum(1 for r in d_results if r["correct"])
            print(f"    {d:<10} {d_passed}/{len(d_results)}")

    divider("═")
    return results

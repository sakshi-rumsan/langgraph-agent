import time

from .logger import AGENT_ICONS, C, divider


class SessionStats:
    def __init__(self):
        self.turn            = 0
        self.agent_calls:  dict[str, int] = {}
        self.tool_calls:   dict[str, int] = {}
        self.handoffs:     list[str]      = []
        self.blocked_handoffs             = 0
        self.start_time                   = time.time()

    def record_agent(self, name: str):
        self.agent_calls[name] = self.agent_calls.get(name, 0) + 1

    def record_tool(self, name: str):
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def record_handoff(self, frm: str, to: str):
        self.handoffs.append(f"{frm} → {to}")

    def record_blocked(self):
        self.blocked_handoffs += 1

    def print_summary(self):
        elapsed = time.time() - self.start_time
        divider("═")
        print(f"{C.BOLD}  📊  SESSION SUMMARY{C.RESET}")
        divider("═")
        print(f"  Total turns          : {self.turn}")
        print(f"  Session time         : {elapsed:.1f}s")
        print(f"  Blocked handoffs     : {C.WARN}{self.blocked_handoffs}{C.RESET}")
        if self.agent_calls:
            print(f"\n  Agent calls:")
            for agent, count in self.agent_calls.items():
                icon = AGENT_ICONS.get(agent, "🤖")
                print(f"    {icon} {agent:<20} {count}x")
        if self.tool_calls:
            print(f"\n  Tool calls:")
            for tool_name, count in self.tool_calls.items():
                print(f"    🔧 {tool_name:<25} {count}x")
        if self.handoffs:
            print(f"\n  Handoffs ({len(self.handoffs)}):")
            for h in self.handoffs:
                print(f"    🔁 {h}")
        # print(f"\n  Log saved to         : {C.SUCCESS}{log_filename}{C.RESET}")
        divider("═")


stats = SessionStats()

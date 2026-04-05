import json
import logging
from datetime import datetime


# ──────────────────────────────────────────────
#  ANSI Color Codes
# ──────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ROUTER  = "\033[95m"   # Magenta
    STUDY   = "\033[94m"   # Blue
    CODING  = "\033[92m"   # Green
    WRITING = "\033[93m"   # Yellow
    TOOL    = "\033[96m"   # Cyan
    HANDOFF = "\033[91m"   # Red
    MSG     = "\033[97m"   # White
    TIME    = "\033[90m"   # Dark gray
    SUCCESS = "\033[32m"   # Dark green
    DIVIDER = "\033[90m"   # Dark gray
    WARN    = "\033[33m"   # Orange/yellow warning


AGENT_COLORS = {
    "router":            C.ROUTER,
    "weather_agent":     C.STUDY,
    "places_agent":      C.CODING,
    "itinerary_agent":   C.WRITING,
}

AGENT_ICONS = {
    "router":            "🔀",
    "weather_agent":     "🌤️ ",
    "places_agent":      "📍",
    "itinerary_agent":   "📋",
}


# ──────────────────────────────────────────────
#  File Logger
# ──────────────────────────────────────────────
# log_filename = f"agent_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

file_logger = logging.getLogger("agent_trace")
file_logger.setLevel(logging.DEBUG)
# _fh = logging.FileHandler(log_filename, encoding="utf-8")
# _fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
# file_logger.addHandler(_fh)


def divider(char="─", width=60, color=C.DIVIDER):
    print(f"{color}{char * width}{C.RESET}")


def log(level: str, agent: str, event: str, detail: str = "", data: dict = None):
    """Central logging — color-coded console + plain file."""
    color = AGENT_COLORS.get(agent, C.MSG)
    icon  = AGENT_ICONS.get(agent, "🤖")
    ts    = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"{C.TIME}[{ts}]{C.RESET} {color}{C.BOLD}{icon} {agent.upper():<14}{C.RESET}"
          f"  {C.BOLD}{level:<12}{C.RESET}  {event}")
    if detail:
        for line in detail.splitlines():
            print(f"         {C.DIM}{line}{C.RESET}")

    file_entry = f"{icon} {agent.upper()} | {level} | {event}"
    if detail:
        file_entry += f"\n         {detail}"
    if data:
        file_entry += f"\n         DATA: {json.dumps(data, default=str)}"
    file_logger.info(file_entry)

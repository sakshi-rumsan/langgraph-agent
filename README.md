# App Langraph

## Overview

This project is a modular Python application designed to leverage LangChain and LangGraph for building multi-agent systems. The application is structured for maintainability and scalability, following best practices in software design.

## Features

- Modular design with clear separation of concerns.
- Support for multiple subagents and tools.
- Configurable via environment variables.
- Logging and session statistics tracking.

## Project Structure

```
app_langraph/
├── src/
│   ├── config.py       # Configuration constants and environment loading
│   ├── logger.py       # Logging utilities
│   ├── stats.py        # Session statistics tracking
│   ├── state.py        # State management
│   ├── subagents.py    # Subagent definitions
│   ├── tools.py        # Tools for subagents and handoffs
│   ├── agents.py       # Main agent definitions
│   ├── router.py       # Router node for message classification
│   └── graph.py        # Graph compilation
├── main.py             # Entry point for the application
├── pyproject.toml      # Project metadata and dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd app_langraph
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To start the application, run:

```bash
python main.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [LangGraph](https://github.com/langgraph/langgraph)

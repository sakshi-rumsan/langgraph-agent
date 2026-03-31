from memory_config import get_memory
from working_memory import get_working_memory
from episodic_memory import add_episodic_memory, search_episodic_memory

memory = get_memory()
messages = get_working_memory()

# Uncomment to add episodic memory
add_episodic_memory(memory, messages, user_id="alice", metadata={"category": "movies"})

results = search_episodic_memory(memory, "What type of movie do i like?", user_id="alice")
print(results)
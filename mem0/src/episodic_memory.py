# Episodic memory operations

def add_episodic_memory(memory, messages, user_id, metadata=None):
    memory.add(messages, user_id=user_id, metadata=metadata)

def search_episodic_memory(memory, query, user_id):
    return memory.search(query, user_id=user_id)

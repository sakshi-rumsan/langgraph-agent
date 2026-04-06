from mem0 import Memory
import os
from dotenv import load_dotenv


CUSTOM_UPDATE_MEMORY_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform three operations: (1) add into the memory, (2) update the memory, and (3) no change.
You MUST NEVER delete any memory. If a new fact contradicts an existing memory, UPDATE it instead of deleting.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element (also use this when facts contradict existing memory)
- NONE: Make no change (if the fact is already present or irrelevant)

Guidelines:

1. **Add**: If the retrieved facts contain new information not present in the memory, add it with a new ID.

2. **Update**: If the retrieved facts contradict or refine existing memory, UPDATE the existing entry.
   Keep the same ID. Merge or replace the old content with the new fact.

3. **No Change**: If the retrieved facts are already present in memory, return NONE.

IMPORTANT: Never use "DELETE" as an event. Always prefer "UPDATE" over "DELETE".
"""



def get_memory_config():
    load_dotenv()
    return {
        "custom_update_memory_prompt": CUSTOM_UPDATE_MEMORY_PROMPT,
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": os.getenv("QDRANT_COLLECTION_NAME"),
            "url":os.getenv("QDRANT_URL"),
   
            "embedding_model_dims": 768,
            "api_key": os.getenv("QDRANT_API_KEY"),
        }
    },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:latest",
                "temperature":  0.1,
                "max_tokens": 2000,
                "top_k": 2,
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            },
        },
    }

def get_memory():
    config = get_memory_config()
    return Memory.from_config(config)


# Episodic memory operations

def add_episodic_memory(memory, messages, user_id, metadata=None):
    memory.add(messages, user_id=user_id, metadata=metadata)

def search_episodic_memory(memory, query, user_id, limit=5):
    """
    Search episodic memory for relevant context.
    
    Args:
        memory: Memory object
        query: Search query
        user_id: User identifier
        limit: Number of results to return (default 5)
    
    Returns:
        Search results with memories
    """
    return memory.search(query, user_id=user_id, limit=limit)

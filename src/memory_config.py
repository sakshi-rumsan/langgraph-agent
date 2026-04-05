from mem0 import Memory
import os
from dotenv import load_dotenv

def get_memory_config():
    load_dotenv()
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "episodic_memory",
                "embedding_model_dims": 768,
                "url": "http://localhost:6333",
                "api_key": "rumsan",
                # "on_disk": True
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1:latest",
                "temperature":  0.1,
                "max_tokens": 2000,
                "top_k": 2,
                "ollama_base_url": "https://jo3m4y06rnnwhaz.askbhunte.com",
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": "https://jo3m4y06rnnwhaz.askbhunte.com",
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

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
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": os.environ.get("OLLAMA_MODEL", "llama3.1:latest"),
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", 0.1)),
                "max_tokens": int(os.environ.get("OLLAMA_MAX_TOKENS", 2000)),
                "ollama_base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            },
        },
    }

def get_memory():
    config = get_memory_config()
    return Memory.from_config(config)

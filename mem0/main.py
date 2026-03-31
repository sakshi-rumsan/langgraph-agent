from services.memory_service import MemoryService
from api.qdrant_api import QdrantAPI
from interfaces.message import Message

# Example chat interface

def main():
    memory = MemoryService()
    qdrant = QdrantAPI()

    # Example: Add a conversation
    messages = [
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
        {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
        {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
        {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
    ]
    memory.add(messages, user_id="alice", metadata={"category": "movies"})

    # Example: Qdrant health check
    print("Qdrant health:", qdrant.health())

if __name__ == "__main__":
    main()

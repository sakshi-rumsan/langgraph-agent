---
name: episodic-memory
description: Store and retrieve user-specific past interactions, preferences, and events. Use when the user refers to past conversations, preferences, or when long-term context is needed.
---

# Episodic Memory Skill

## When to use

- User says: "last time", "earlier", "previously"
- Personalization needed
- Long-term context required

## Add Memory

Store important interactions:

- User preferences
- Decisions
- Key events
- Corrections

## Retrieve Memory

Search memory when:

- Query relates to past
- Personalization required

## Strategy

1. Detect importance
2. Store concise summary
3. Retrieve top-k relevant memories
4. Inject into reasoning context

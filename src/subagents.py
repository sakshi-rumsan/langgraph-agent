from langchain.agents import create_agent

from .config import MODEL

weather_subagent = create_agent(
    model=MODEL,
    name="weather_subagent",
    system_prompt=(
        "You are a weather expert for travel planning. Provide detailed weather analysis for a location. "
        "Include temperature, conditions, and travel recommendations. "
        "If rainy, suggest bringing an umbrella. If heavy rain, advise against outdoor activities. "
        "Keep responses concise but informative."
    ),
)

places_subagent = create_agent(
    model=MODEL,
    name="places_subagent",
    system_prompt=(
        "You are a travel destination expert. Recommend popular attractions, restaurants, and activities "
        "for a location. Provide brief descriptions and why they're worth visiting. "
        "Organize by category (landmarks, dining, activities). Keep responses well-organized."
    ),
)

itinerary_subagent = create_agent(
    model=MODEL,
    name="itinerary_subagent",
    system_prompt=(
        "You are a travel itinerary planner. Create a complete day-trip itinerary with timing. "
        "Include recommended activities from morning to evening, travel times between locations, "
        "and weather-appropriate suggestions (e.g., indoor activities if rainy). "
        "Structure: 09:00 AM - 05:00 PM with break times. Be practical and realistic."
    ),
)


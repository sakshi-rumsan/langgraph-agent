import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not APIFY_API_TOKEN:
    raise ValueError("APIFY_API_TOKEN is not set in the environment variables.")

from dotenv import load_dotenv

load_dotenv()

MODEL = "openai:gpt-4o-mini"
MAX_HANDOFFS = 2
MAX_TURNS = 10

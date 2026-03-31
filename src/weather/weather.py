import requests
from src.config import GEOCODING_API_URL, WEATHER_API_URL, OPENWEATHER_API_KEY


API_KEY = OPENWEATHER_API_KEY   # 🔑 replace this


def get_weather_by_location(location_name: str):
    # 1. Geocoding: location → lat/lon
    geo_params = {
        "q": location_name,
        "limit": 1,
        "appid": API_KEY
    }

    geo_res = requests.get(GEOCODING_API_URL, params=geo_params).json()

    if not geo_res:
        return {"error": "Location not found"}

    lat = geo_res[0]["lat"]
    lon = geo_res[0]["lon"]
    name = geo_res[0]["name"]
    country = geo_res[0]["country"]

    # 2. Weather API
    weather_params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"   # Celsius
    }

    weather_res = requests.get(WEATHER_API_URL, params=weather_params).json()

    # 3. Clean JSON response
    result = {
        "location": {
            "name": name,
            "country": country,
            "lat": lat,
            "lon": lon
        },
        "weather": {
            "temperature": weather_res["main"]["temp"],
            "feels_like": weather_res["main"]["feels_like"],
            "humidity": weather_res["main"]["humidity"],
            "pressure": weather_res["main"]["pressure"],
            "weather": weather_res["weather"][0]["description"],
            "wind_speed": weather_res["wind"]["speed"]
        }
    }

    return result
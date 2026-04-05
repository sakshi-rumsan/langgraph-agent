import os
import hashlib
from apify_client import ApifyClient
from src.config import APIFY_API_TOKEN

# Simple in-memory cache for search results
_places_cache = {}

def fetch_places_from_apify(location, search_strings, max_places=5, language="en", skip_cache=False):
    """
    Fetch popular places from Apify based on location and search criteria.
    Results are cached to avoid re-fetching the same location+categories.

    Args:
        location (str): The location to search for places (e.g., "New York, USA").
        search_strings (list): List of search strings (e.g., ["restaurant", "cafe"]).
        max_places (int): Maximum number of places to fetch per search. Default is 5.
        language (str): Language for the results. Default is "en".
        skip_cache (bool): Skip cache and fetch fresh results.

    Returns:
        list: A list of places with details fetched from Apify.
    """
    # Check cache first
    cache_key = hashlib.md5(f"{location}:{'|'.join(sorted(search_strings))}".encode()).hexdigest()
    if cache_key in _places_cache and not skip_cache:
        return _places_cache[cache_key]
    
    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API_TOKEN)

    # Prepare the Actor input
    run_input = {
    "searchStringsArray": search_strings,
    "locationQuery": location,
    "maxCrawledPlacesPerSearch": 3,
    "language": "en",
    "categoryFilterWords": [],
    "searchMatching": "all",
    "placeMinimumStars": "",
    "website": "allPlaces",
    "skipClosedPlaces": False,
    "scrapePlaceDetailPage": False,
    "scrapeTableReservationProvider": False,
    "includeWebResults": False,
    "scrapeDirectories": False,
    "maxQuestions": 0,
    "scrapeContacts": False,
    "scrapeSocialMediaProfiles": {
        "facebooks": False,
        "instagrams": False,
        "youtubes": False,
        "tiktoks": False,
        "twitters": False,
    },
    "maximumLeadsEnrichmentRecords": 0,
    "leadsEnrichmentDepartments": [],
    "maxReviews": 0,
    # "reviewsStartDate": "",
    "reviewsSort": "newest",
    "reviewsFilterString": "",
    "reviewsOrigin": "all",
    "scrapeReviewsPersonalData": True,
    "maxImages": 0,
    "scrapeImageAuthors": False,
    # "countryCode": None,
    # "city": None,
    # "state": None,
    # "county": None,
    # "postalCode": None,
    # "customGeolocation": None,
    # "startUrls": None,
    # "placeIds": None,
    "allPlacesNoSearchAction": "",
}


    # Run the Actor and wait for it to finish
    run = client.actor("nwua9Gu5YrADL7ZDj").call(run_input=run_input)

    # Fetch and return Actor results from the run's dataset
    results = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        results.append(item)
    
    # Cache the results
    _places_cache[cache_key] = results
    
    return results


def smart_search_categories(query):
    """
    Intelligently choose which categories to search based on user's query.
    Reduces search time by only querying relevant categories.
    """
    query_lower = query.lower()
    
    # Food-focused queries
    if any(word in query_lower for word in ["restaurant", "food", "eat", "dining", "cafe", "vegan", "vegetarian"]):
        return ["restaurants"]
    
    # Sightseeing-focused queries
    if any(word in query_lower for word in ["attraction", "landmark", "sight", "temple", "museum", "monument"]):
        return ["attractions", "landmarks"]
    
    # Shopping/shopping district queries
    if any(word in query_lower for word in ["shop", "market", "district", "mall"]):
        return ["attractions", "landmarks"]
    
    # Default: search all categories
    return ["attractions", "restaurants", "landmarks"]

# Example usage
# if __name__ == "__main__":

#     LOCATION = "New York, USA"
#     SEARCH_STRINGS = ["restaurant"]

#     try:
#         places = fetch_places_from_apify( LOCATION, SEARCH_STRINGS)
#         print("Fetched Places:")
#         for place in places:
#             print(place)
#     except Exception as e:
#         print(f"Error: {e}")
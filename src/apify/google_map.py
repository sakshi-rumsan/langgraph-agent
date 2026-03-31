import os
from apify_client import ApifyClient
from src.config import APIFY_API_TOKEN

def fetch_places_from_apify(location, search_strings, max_places=5, language="en"):
    """
    Fetch popular places from Apify based on location and search criteria.

    Args:
        api_token (str): Your Apify API token.
        location (str): The location to search for places (e.g., "New York, USA").
        search_strings (list): List of search strings (e.g., ["restaurant", "cafe"]).
        max_places (int): Maximum number of places to fetch per search. Default is 50.
        language (str): Language for the results. Default is "en".

    Returns:
        list: A list of places with details fetched from Apify.
    """
    # Initialize the ApifyClient with your API token
    client = ApifyClient(APIFY_API_TOKEN)

    # Prepare the Actor input
    run_input = {
    "searchStringsArray": search_strings,
    "locationQuery": location,
    "maxCrawledPlacesPerSearch":5,
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
    # print(f"Fetched {len(results)} places from Apify for location: {results}")
    
    return results

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
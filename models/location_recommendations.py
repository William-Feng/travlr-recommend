import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests

NUM_CLASSES = 11

def get_preferences():
    url = "http://localhost:4000/locations"
    response = requests.get(url)
    if response.status_code == 200:
        locations = response.json()
        return locations
    else:
        print("Error: Failed to retrieve the locations.")
        return None


def data_ingestion():
    data = {
        'attraction': ['Sydney Opera House', 'Bondi Beach', 'The Rocks', 'Royal Botanic Garden Sydney', 'Sydney Harbour Bridge', 'Darling Harbour', 'Taronga Zoo', 'Blue Mountains', 'Hunter Valley Gardens', 'Jenolan Caves'],
        'food': [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
        'nature': [0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
        'adventure': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'culture': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
        'exercise': [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        'tourist_hotspot': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cozy': [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        'family': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        'wildlife': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        'near_cbd': [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        'disabled_accessibility': [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
    }
    data = pd.DataFrame(data)
    return data

def data_to_json(data):
    json_data = data.to_json()
    return json_data

class UserBasedCollaborativeFiltering:
    def __init__(self, json_data):
        self.data = pd.read_json(json_data)
        self.data = self.data.set_index("id")
        self.user_similarity = cosine_similarity(self.data)

    def get_similar_locations(self, id, top_n=5):
        user_idx = self.data.index.get_loc(id)
        user_scores = list(enumerate(self.user_similarity[user_idx]))
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [self.data.index[user]
                         for user, _ in user_scores[1:top_n+1]]

        return similar_users


def run_location_recommendation_system(location_id):
    data = get_preferences()
    data = json.dumps(data)
    data = pd.read_json(data)
    data = data.drop(["name", "photo_url", "coordinates"], axis=1)
    json_data = data.to_json()

    cf = UserBasedCollaborativeFiltering(json_data)
    print("HELLO")
    print(location_id)
    print("HELLO")
    similar_locations = cf.get_similar_locations(id=location_id, top_n=10)

    # Return the most similar people in an array (from most similar to least similar)
    result = []
    counter = 1
    for location in similar_locations:
        print(f"The location that is {counter} similar to {location_id} has location_id: {location}")
        result.append(location)
        counter += 1
        
    return result
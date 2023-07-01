import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests

NUM_CLASSES = 11

def get_preferences():
    url = "http://localhost:8000/locations"
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

def get_name():
    name = 'Palm Beach'
    return name

def data_to_json(data):
    json_data = data.to_json()
    return json_data


# class UserBasedCollaborativeFiltering:
#     def __init__(self, json_data):
#         df = pd.read_json(json_data)
#         self.data = df.set_index("attraction")
#         self.user_similarity = cosine_similarity(self.data)

#     def get_similar_users(self, user_id, top_n=5):
#         user_scores = list(enumerate(self.user_similarity[user_id]))
#         user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
#         similar_users = [user for user, _ in user_scores[1:top_n+1]]

#         return similar_users

class UserBasedCollaborativeFiltering:
    def __init__(self, json_data):
        self.data = pd.read_json(json_data)
        self.data = self.data.set_index("name")
        #   user_ids = self.data["user_id"]
        # self.data = self.data.drop(["user_id"], axis = 1)
        # self.data_copy = pd.read_json(json_data)
        self.user_similarity = cosine_similarity(self.data)

    def get_similar_users(self, name, top_n=5):
        user_idx = self.data.index.get_loc(name)
        # print(user_idx)
        user_scores = list(enumerate(self.user_similarity[user_idx]))
        # print(user_scores)
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [self.data.index[user] for user, _ in user_scores[1:top_n+1]]

        return similar_users


if __name__ == "__main__":
    # data = data_ingestion()
    data = get_preferences()
    data = json.dumps(data)
    data = pd.read_json(data)
    data = data.drop(["id", "photo_url", "coordinates"], axis = 1)
    #print(data)
    json_data = data.to_json()
    
    # json_data = data_to_json(data)
    # filtering = UserBasedCollaborativeFiltering(json_data)
    # place_id = len(data) - 1
    # similar_users = filtering.get_similar_users(place_id)
    # counter = 1
    # for value in similar_users:
    #     print(
    #         f"The place that is {counter} similar to {place_id} has id: {value}")
    #     counter += 1
    
    cf = UserBasedCollaborativeFiltering(json_data)
    similar_locations = cf.get_similar_users(name='Palm Beach', top_n=3)
    name = get_name()
    counter = 1
    for location in similar_locations:
        print(f"Location {counter} similar to {name} is: {location}")
        counter += 1 

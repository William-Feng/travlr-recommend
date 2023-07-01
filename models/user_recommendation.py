import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json


'''
This script is: 
1. Calling the PyTorch model
2. Getting the vector of what's inside the photo 
3. Aggregating the value 
4. Using recommendation system to get the people most similar to the person uploading the photos! 
'''

NUM_CLASSES = 11

def get_preferences():
    url = "http://localhost:4000/preferences"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        preferences = response.json()
        return preferences
    else:
        print("Error: Failed to retrieve user preferences.")
        return None




def data_ingestion():
    data = pd.DataFrame({'user_id': [0, 1, 2, 3, 1, 3],
                        'item1': [5, 2, 1, 4, 1, 3],
                         'item2': [4, 4, 2, 3, 2, 3],
                         'item3': [3, 5, 4, 2, 3, 3],
                         'item4': [1, 2, 5, 1, 3, 3]})
    data = data.groupby('user_id').mean()
    json_data = data.to_json()
    return json_data


class UserBasedCollaborativeFiltering:
    def __init__(self, json_data):
        self.data = pd.read_json(json_data)
        self.data = self.data.set_index("user_id")
        self.user_similarity = cosine_similarity(self.data)

    def get_similar_users(self, user_id, top_n=5):
        user_idx = self.data.index.get_loc(user_id)
        user_scores = list(enumerate(self.user_similarity[user_idx]))
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [self.data.index[user]
                         for user, _ in user_scores[1:top_n+1]]

        return similar_users


def get_userid():
    userid = '4c6376e4-a587-4ce7-b588-6a94ab103685'
    return userid


def run_user_recommendation_system(user_id):
    data = get_preferences()
    data = json.dumps(data)
    data = pd.read_json(data)
    data = data.drop(["id"], axis=1)
    json_data = data.to_json()

    cf = UserBasedCollaborativeFiltering(json_data)
    similar_users = cf.get_similar_users(
        user_id=user_id, top_n=10)

    # Return the most similar people in an array (from most similar to least similar)
    result = []
    counter = 1
    for person in similar_users:
        print(f"The person who is {counter} similar has user_id: {person}")
        result.append(person)
        counter += 1
        

    url = "http://127.0.0.1:4000/connections"
    
    data = {
        "user_id": user_id,
        "connection_ids": result
    }
    
    print(json.dumps(data))
    
    response = requests.post(url, json=json.dumps(data))
    
    return result

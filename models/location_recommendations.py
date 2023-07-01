import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from io import BytesIO

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

def get_photos():
    url = "http://localhost:4000/photos"
    response = requests.get(url)
    if response.status_code == 200:
        preferences = response.json()
        return preferences
    else:
        print("Error: Failed to retrieve user preferences.")
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
        
        print("HELLO")
        print(self.data)
        print("HELLO")
        
        self.data = self.data.set_index("id")
        self.user_similarity = cosine_similarity(self.data)

    def get_similar_locations(self, id, top_n=5):
        user_idx = self.data.index.get_loc(id)
        user_scores = list(enumerate(self.user_similarity[user_idx]))
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [self.data.index[user]
                         for user, _ in user_scores[1:top_n+1]]

        return similar_users


class FineTunedModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FineTunedModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x


def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def image_classification(photo, num_classes=NUM_CLASSES):
    # Call PyTorch Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    model = FineTunedModel(num_classes)
    model.load_state_dict(torch.load('models/model_learning_curve.pth'))

    # image = Image.open('models/test/lunar_park.jpg').convert('RGB')
    image = load_image_from_url(photo).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output)
        predicted_labels = (probabilities >= 0.35).squeeze().tolist()
        
    print(predicted_labels)

    return predicted_labels

def run_location_recommendation_system(user_id):
    data = get_preferences()
    data = json.dumps(data)
    data = pd.read_json(data)
    data = data.drop(["name", "photo_url", "coordinates"], axis=1)
    json_data = data.to_json()
    
    url = f"http://127.0.0.1:4000/photos/user/{user_id}"
    
    response = requests.get(url)
    
    agg = []
    
    for res in response:
        stats = image_classification(photo=res["url"])
        agg.append(list(map(int, stats)))
        
    averages = [sum(col) / len(col) for col in zip(*agg)]
        
    new_row = pd.DataFrame([averages]).T
    
    print(new_row)

    # Add the new row to the dataframe
    df = pd.concat([data, new_row], axis=1)
    
    location_url = "http://127.0.0.1:4000/locations"
    locations_response = requests.get(location_url)
    
    for location in locations_response:
        del location["name"]
        del location["photo_url"]
        del location["coordinates"]
        
    print(location)
    
    
    # for res in response:
    #     stats = image_classification(photo=res["url"])
    #     agg.append(list(map(int, stats)))
    
    
    # print(df)
        
    # url = "http://127.0.0.1:4000/locations"
    
    # data = {
    #     "user_id": user_id,
    #     "recommendation_id": averages
    # }
    
    # json_data = df.to_json()
    
    
    # print(json.dumps(data))
    
    # response = requests.post(url, json=json.dumps(data))

    # cf = UserBasedCollaborativeFiltering(json_data)
    # similar_locations = cf.get_similar_locations(id=location_id, top_n=10)

    # Return the most similar people in an array (from most similar to least similar)
    # result = []
    # counter = 1
    # for location in similar_locations:
    #     print(f"The location that is {counter} similar to {location_id} has location_id: {location}")
    #     result.append(location)
    #     counter += 1
        
    # url = "http://127.0.0.1:4000/recommendations"
    
    # data = {
    #     "user_id": user_id,
    #     "location_id": result
    # }
    
    # print(json.dumps(data))
    
    # response = requests.post(url, json=json.dumps(data))
        
        
    # return result
    
    
# model = FineTunedModel(NUM_CLASSES)
# model.load_state_dict(torch.load('model_learning_curve.pth'))
# image = Image.open("image.jpg").convert('RGB')
# image = transform(image)
# image = image.unsqueeze(0)  # Add an extra dimension for batch size

# # Make predictions
# with torch.no_grad():
#     output = model(image)
#     probabilities = torch.sigmoid(output)
#     predicted_labels = (probabilities >= 0.4).squeeze().tolist()

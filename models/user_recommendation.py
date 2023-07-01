import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

'''
This script is: 
1. Calling the PyTorch model
2. Getting the vector of what's inside the photo 
3. Aggregating the value 
4. Using recommendation system to get the people most similar to the person uploading the photos! 
'''

NUM_CLASSES = 11

def get_preferences():
    url = "http://localhost:8000/preferences"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        preferences = response.json()
        return preferences
    else:
        print("Error: Failed to retrieve user preferences.")
        return None
    
# def get_photos():
#     url = "http://localhost:8000/photos"
#     response = requests.get(url)
#     print(response)
#     if response.status_code == 200:
#         preferences = response.json()
#         return preferences
#     else:
#         print("Error: Failed to retrieve user preferences.")
#         return None


def data_ingestion():
    data = pd.DataFrame({'user_id': [0, 1, 2, 3, 1, 3],
                        'item1': [5, 2, 1, 4, 1, 3],
                         'item2': [4, 4, 2, 3, 2, 3],
                         'item3': [3, 5, 4, 2, 3, 3],
                         'item4': [1, 2, 5, 1, 3, 3]})
    data = data.groupby('user_id').mean()
    # json_data = get_preferences()
    json_data = data.to_json()
    return json_data


class UserBasedCollaborativeFiltering:
    def __init__(self, json_data):
        self.data = pd.read_json(json_data)
        self.data = self.data.set_index("user_id")
        #   user_ids = self.data["user_id"]
        # self.data = self.data.drop(["user_id"], axis = 1)
        # self.data_copy = pd.read_json(json_data)
        self.user_similarity = cosine_similarity(self.data)

    def get_similar_users(self, user_id, top_n=5):
        user_idx = self.data.index.get_loc(user_id)
        # print(user_idx)
        user_scores = list(enumerate(self.user_similarity[user_idx]))
        # print(user_scores)
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [self.data.index[user] for user, _ in user_scores[1:top_n+1]]

        return similar_users


def get_userid():
    userid = '4c6376e4-a587-4ce7-b588-6a94ab103685'
    return userid


# class FineTunedModel(nn.Module):
#     def __init__(self, num_classes):
#         super(FineTunedModel, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.resnet(x)
#         return x

class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
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


def image_classification(num_classes=NUM_CLASSES):
    # Call PyTorch Model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    model = FineTunedModel(num_classes)
    model.load_state_dict(torch.load('model_learning_curve.pth'))

    image = Image.open('./test/lunar_park.jpg').convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output)
        predicted_labels = (probabilities >= 0.35).squeeze().tolist()

    return predicted_labels


if __name__ == "__main__":
    # json_data = data_ingestion()
    # filtering = UserBasedCollaborativeFiltering(json_data)
    # user_id = get_userid()
    # similar_users = filtering.get_similar_users(user_id)

    # counter = 1
    # for person in similar_users:
    #     print(f"The person who is {counter} similar has userid: {person}")
    #     counter += 1

    # print(image_classification())
    data = get_preferences()
    data = json.dumps(data)
    data = pd.read_json(data)
    data = data.drop(["id"], axis = 1)
    json_data = data.to_json()
    
    cf = UserBasedCollaborativeFiltering(json_data)
    similar_users = cf.get_similar_users(user_id='4c6376e4-a587-4ce7-b588-6a94ab103685', top_n=5)
    counter = 1
    for person in similar_users:
        print(f"The person who is {counter} similar has userid: {person}")
        counter += 1
    
    photos = get_photos()
    photos = json.dumps(photos)
    photos = pd.read_json(photos)
    # data = data.drop(["id"], axis = 1)
    print(photos)



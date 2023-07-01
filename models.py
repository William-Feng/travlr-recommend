import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from finetune import FineTunedModel

# def finetune_predict(num_classes = 9):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),     # Resize to a common size
#         transforms.ToTensor(),              # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
#     ])
#     model = FineTunedModel(num_classes)
#     model.load_state_dict(torch.load('model.pth'))

#     image = Image.open('./test/lunar_park.jpg').convert('RGB')
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add an extra dimension for batch size

#     # Make predictions
#     with torch.no_grad():
#         output = model(image)
#         probabilities = torch.sigmoid(output)
#         predicted_labels = (probabilities >= 0.5).squeeze().tolist()
#     return predicted_labels

# print(finetune_predict(num_classes=9))

class ItemBasedCollaborativeFiltering:
    '''
    CSESoc Flagship Hackathon. 
    We utilised Collaborative Filtering for items to attain the top recommendations for a user. 
    '''
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.item_similarity = cosine_similarity(self.data.T)
       
    def get_similar_items(self, item_id, top_n=5):
        '''
        Input: 
            item_id: 
            top_n: The top recommended results (default = 5)
        Output:
            similar_items: 
        '''
        item_scores = list(enumerate(self.item_similarity[item_id]))
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        similar_items = [item for item, _ in item_scores[1:top_n+1]]
        return similar_items
    
    def get_recommendations(self, user_id, top_n=5):
        user_ratings = self.data.loc[user_id].tolist()
        unrated_items = [i for i, rating in enumerate(user_ratings) if pd.isnull(rating)]
        item_scores = []
        
        for item_id in unrated_items:
            similar_items = self.get_similar_items(item_id)
            rating_sum = 0
            weight_sum = 0
            
            for similar_item in similar_items:
                if not pd.isnull(user_ratings[similar_item]):
                    rating_sum += user_ratings[similar_item] * self.item_similarity[item_id, similar_item]
                    weight_sum += self.item_similarity[item_id, similar_item]
            
            if weight_sum > 0:
                item_scores.append((item_id, rating_sum / weight_sum))
        
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in item_scores[:top_n]]
        return recommended_items

class UserBasedCollaborativeFiltering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.user_similarity = cosine_similarity(self.data)
    
    def get_similar_users(self, user_id, top_n=5):
        user_scores = list(enumerate(self.user_similarity[user_id]))
        user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
        similar_users = [user for user, _ in user_scores[1:top_n+1]]
        return similar_users

# Usage for ItemBasedCollaborativeFiltering
filtering = ItemBasedCollaborativeFiltering('ratings.csv')
user_id = 1
recommendations = filtering.get_recommendations(user_id)
print(f"Recommended items for user {user_id}:")
for item in recommendations:
    print(item)
    
# Usage for UserBasedCollaborativeFiltering
filtering = UserBasedCollaborativeFiltering('ratings.csv')
user_id = 1
similar_users = filtering.get_similar_users(user_id)
print(f"Most similar users to user {user_id}:")
for user in similar_users:
    print(user)

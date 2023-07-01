import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[index]
        
        return image, label

# Define the CNN model with fine-tuning
class FineTunedModel(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the paths to your images and corresponding labels
image_paths = ['./images/operahouse.jpg', './images/bondi_beach.jpg', './images/the_rocks.jpg',
       './images/royal_botanic_garden.jpg', './images/sydney_harbour_bridge.jpg',
       './images/darling_harbour.jpg', './images/taronga_zoo.jpg', './images/blue_mountains.jpg',
       './images/hunter_valley_gardens.jpg', './images/jenolan_caves.jpg']
labels = torch.tensor([
    [0, 0, 0, 1, 0, 1, 0, 1, 0],  # Label for image1
    [1, 1, 1, 0, 1, 1, 0, 1, 1],  
    [1, 0, 0, 1, 1, 1, 1, 1, 0], 
    [1, 1, 0, 1, 1, 1, 1, 1, 1], 
    [0, 0, 1, 0, 0, 1, 0, 0, 0], 
    [1, 0, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 1, 0, 1, 1], 
    [0, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 0, 0, 1, 1, 1, 1, 0], 
    [0, 1, 1, 0, 0, 1, 0, 1, 1]
], dtype=torch.float)

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),     # Resize to a common size
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

# Create the dataset
dataset = CustomDataset(image_paths, labels, transform=transform)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model with fine-tuning
num_classes = 9
model = FineTunedModel(num_classes)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Print the loss after each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        predicted_labels = (torch.sigmoid(outputs) >= 0.5).float()
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100}%')



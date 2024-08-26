import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
import matplotlib.pyplot as plt

# Path to the directory containing the images and .cat files
data_dir = "CAT_DATASET_01/CAT_00"

# Function to load an image and its corresponding .cat file
def load_image_and_annotation(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    except Exception as e:
        print(f"Error loading image: {image_path} - {e}")
        return None, None
    
    annotation_path = image_path + ".cat"
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found for image: {image_path}")
        return None, None

    try:
        with open(annotation_path, "r") as f:
            annotation = list(map(int, f.read().split()[1:]))  # Skip the first value, which is the number of points
        annotation = np.array(annotation).reshape(-1, 2)
    except Exception as e:
        print(f"Error loading annotation: {annotation_path} - {e}")
        return None, None

    return image, annotation

# Get all image paths in the directory, only including those with corresponding .cat files
image_paths = [
    os.path.join(data_dir, fname) 
    for fname in os.listdir(data_dir) 
    if fname.endswith(".jpg") and os.path.exists(os.path.join(data_dir, fname + ".cat"))
]

# Initialize the image processor for the Vision Transformer
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# Create a dataset and data loader
class CatDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image, annotation = load_image_and_annotation(image_path)
        
        if image is None or annotation is None:
            return None, None

        inputs = self.processor(images=image, return_tensors="pt")
        targets = torch.tensor(annotation.flatten(), dtype=torch.float)
        
        # Normalize the targets to be between 0 and 1
        image_size = 224  # Assuming images are resized to 224x224
        targets = targets / image_size

        return inputs["pixel_values"].squeeze(0), targets

dataset = CatDataset(image_paths, processor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

# Load the pre-trained Vision Transformer model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
model.classifier = nn.Linear(model.config.hidden_size, 18)  # 9 points, each with x and y coordinates

# Initialize the classifier weights
nn.init.xavier_uniform_(model.classifier.weight)
nn.init.zeros_(model.classifier.bias)

# Move the model to the GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in data_loader:
        if inputs is None or targets is None:
            continue
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=inputs)
        loss = criterion(outputs.logits, targets)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader)}")

# Save the trained model
model.save_pretrained("vit_facial_feature_model")

# Example of evaluating and visualizing predictions
model.eval()
with torch.no_grad():
    for image_path in image_paths[:1]:
        image, annotation = load_image_and_annotation(image_path)
        if image is None or annotation is None:
            continue
        
        inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        outputs = model(pixel_values=inputs)
        predicted_coords = outputs.logits.cpu().detach().numpy().reshape(-1, 2)

        # Denormalize the predicted coordinates
        predicted_coords = predicted_coords * 224

        plt.imshow(image)
        plt.scatter(annotation[:, 0], annotation[:, 1], color='red', label='True Annotations')
        plt.scatter(predicted_coords[:, 0], predicted_coords[:, 1], color='blue', label='Predicted Points')
        plt.title(f"Annotations and Predictions for {os.path.basename(image_path)}")
        plt.legend()
        plt.show()

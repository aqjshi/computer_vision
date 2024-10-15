import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive image creation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Function to read voxel data from a file
def read_voxel_data(file_path):
    """
    Read voxel data from a file and return as a list of numpy arrays.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            return []
        
        return [np.array(list(map(float, line.strip().split()))) for line in lines]

# Function to preprocess a limited number of voxel files in the specified directory
def preprocess_limited_voxels(directory, num_voxels):
    """
    Preprocess a limited number of voxel files in the specified directory.
    """
    files = sorted(os.listdir(directory))[:num_voxels]
    voxel_list = []

    for file in files:
        file_path = os.path.join(directory, file)
        voxel_data = read_voxel_data(file_path)
        
        if voxel_data:
            voxel_list.append(voxel_data[0])  # Use the first voxel in the file
        else:
            print(f"Warning: Skipped file {file_path} due to empty or malformed data.")
    
    return voxel_list

# Function to preprocess data
def preprocess_data(data_dir, num_voxels):
    """
    Preprocess a specified number of voxel data files from the directory.
    """
    print("Starting voxel data preprocessing...")
    voxel_data = preprocess_limited_voxels(data_dir, num_voxels)
    
    # Pad the data to ensure each has 2187 values
    padded_voxel_data = np.array([np.pad(voxel, (0, 2187 - len(voxel)), mode='constant') for voxel in voxel_data])
    
    print(f"Preprocessing complete. Number of voxel images loaded: {len(padded_voxel_data)}")
    return padded_voxel_data

# Function to preprocess data and generate labels
def preprocess_data_with_labels(data_dir, num_voxels, classification_task):
    voxel_data = preprocess_data(data_dir, num_voxels)
    filtered_voxel_data = []
    labels = []
    unique_classes = set()
    
    files = sorted(os.listdir(data_dir))[:num_voxels]
    
    for i, file in enumerate(files):
        parts = file.split('$')
        chiral_length = int(parts[1])  # chiral_length is the second element in the filename
        rs = parts[2]
        posneg = int(parts[3].split('.')[0])
        
        # Classification tasks
        if classification_task == 0:
            # Task 0: Cast chiral_length > 0 to 1, keep 0
            class_label = 1 if chiral_length > 0 else 0

        elif classification_task == 1:
            # Task 1: 0 vs. 1 Chiral Center
            if chiral_length >= 2:
                continue  # Skip files where chiral length >= 2
            class_label = chiral_length  # 0 if no chiral center, 1 if exactly one

        elif classification_task == 2:
            # Task 2: Number of Chiral Centers
            class_label = chiral_length  # Use the exact number of chiral centers as the label

        elif classification_task == 3:
            # Task 3: R vs. S
            if chiral_length != 1:
                continue  # Skip files where chiral length != 1
            class_label = rs  # 'R' or 'S'

        elif classification_task == 4:
            # Task 4: + vs. - for chiral_length == 1
            if chiral_length != 1:
                continue  # Skip files where chiral length != 1
            class_label = '+' if posneg >= 0 else '-'

        elif classification_task == 5:
            # Task 5: + vs. - for all chiral lengths
            class_label = '+' if posneg >= 0 else '-'

        # Add the class label and corresponding voxel data
        filtered_voxel_data.append(voxel_data[i])  # Append voxel data that passes the filters
        labels.append(class_label)
        unique_classes.add(class_label)
    
    # Map unique classes to numeric labels
    class_to_label = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    numeric_labels = [class_to_label[label] for label in labels]
    
    return filtered_voxel_data, numeric_labels, class_to_label

# Function to convert voxel data to image
def voxel_to_image(voxel_data):
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.cpu().numpy()  # Move to CPU and then convert to numpy array

    r_channel = voxel_data[0::3].reshape((9, 9, 9))  # Red channel
    g_channel = voxel_data[1::3].reshape((9, 9, 9))  # Green channel
    b_channel = voxel_data[2::3].reshape((9, 9, 9))  # Blue channel

    # Normalize the channels to [0, 255]
    r_channel = (r_channel * 255).astype(np.uint8)
    g_channel = (g_channel * 255).astype(np.uint8)
    b_channel = (b_channel * 255).astype(np.uint8)

    # Flatten the 3D grid into a 2D 27x27 image
    r_flat = r_channel.flatten().reshape(27, 27)
    g_flat = g_channel.flatten().reshape(27, 27)
    b_flat = b_channel.flatten().reshape(27, 27)

    # Stack the channels to form a 27x27 RGB image
    image = np.stack((r_flat, g_flat, b_flat), axis=-1)
    
    return image

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Use Conv2d to extract patches and project to embed_dim
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    # x shape: [batch_size, in_channels, img_size, img_size]
    def forward(self, x):
        x = self.proj(x)  # Shape: [batch_size, embed_dim, num_patches_root, num_patches_root]
        x = x.flatten(2)  # Shape: [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        return x
# Custom Transformer Encoder Layer
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(d_model, nhead, **kwargs)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

# Positional encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.create_positional_encoding(embed_dim, num_patches)
    
    def create_positional_encoding(self, embed_dim, num_patches):
        pe = torch.zeros(1, num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        positional_encoding = self.positional_encoding.expand(x.size(0), -1, -1).to(x.device)
        x = x + positional_encoding  # Add positional encoding
        return x

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        self.positional_encoding = PositionalEncoding(embed_dim, self.num_patches)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        embeddings = self.patch_embedding(x)
        embeddings = self.positional_encoding(embeddings)
        
        attention_weights = []
        x = embeddings
        for layer in self.encoder_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        classification_output = self.classifier(x.mean(dim=1))  # Global average pooling and classification
        return classification_output, embeddings, attention_weights

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_predictions = []
        for patches, labels in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Collect predictions for training data
            _, predicted = torch.max(outputs.data, 1)
            all_train_predictions.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        
        # Calculate F1 score on training data
        train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Training F1 Score: {train_f1:.4f}")
        
        # Evaluate on test set and report F1 score
        test_loss, test_accuracy, test_f1 = evaluate_with_f1(model, test_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}")

# Evaluation function that calculates loss, accuracy, and F1 score
def evaluate_with_f1(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for patches, labels in test_loader:
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = running_loss / len(test_loader)
    
    return avg_loss, accuracy, f1

# Function to visualize predictions
def visualize_predictions(voxel_data, labels, predictions, class_to_label, num_images=5):
    label_to_class = {v: k for k, v in class_to_label.items()}  # Reverse the class_to_label mapping
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        voxel = voxel_data[i]
        label = labels[i].item()
        prediction = predictions[i]
        
        image = voxel_to_image(voxel)
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"True: {label_to_class[label]}\nPred: {label_to_class[prediction]}")
    
    plt.suptitle("Model Predictions")
    plt.savefig('model_predictions.png')
    plt.close()

# Evaluation function with prediction visualization, F1 score, and confusion matrix
def evaluate_and_visualize_model(model, dataloader, criterion, voxel_data, y_test, class_to_label, num_images=5):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for patches, labels in dataloader:
            outputs, _, _ = model(patches)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"Final Test Loss: {avg_loss:.4f}, Final Test Accuracy: {accuracy:.2f}%")

    # Calculate F1 score
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"Final Test F1 Score (Weighted): {f1:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(class_to_label.keys()), yticklabels=list(class_to_label.keys()))
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Visualize predictions
    visualize_predictions(voxel_data=voxel_data, labels=y_test, predictions=predictions, class_to_label=class_to_label, num_images=num_images)



if __name__ == "__main__":
    device = torch.device("mps")
    num_voxels = 1000000  # Adjust as needed
    classification_task = 3  # Choose the classification task

    # Preprocess the train data and generate labels
    train_dir = "train_rs/"
    test_dir = "test_rs/"
    
    # Preprocess training and testing data based on the task
    train_voxel_data, train_labels, train_class_to_label = preprocess_data_with_labels(train_dir, num_voxels, classification_task)
    test_voxel_data, test_labels, test_class_to_label = preprocess_data_with_labels(test_dir, num_voxels, classification_task)

    # Ensure train and test class mappings match
    if train_class_to_label != test_class_to_label:
        raise ValueError("Class mappings between training and testing datasets do not match!")

    X_train, y_train = train_voxel_data, train_labels
    X_test, y_test = test_voxel_data, test_labels
    class_to_label = train_class_to_label

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")
    print(f"Number of classes: {len(class_to_label)}")
    print(f"Classification task: {classification_task}")

    # Convert to PyTorch tensors and create DataLoader
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    # Ensure correct image size and reshaping
    img_size = 27  # Image is 27x27 pixels
    in_channels = 3  # RGB channels
    patch_size = 16 # You can experiment with patch sizes

    # Reshape the data to match the expected input for the transformer model
    X_train = X_train.view(-1, in_channels, img_size, img_size)
    X_test = X_test.view(-1, in_channels, img_size, img_size)

    # Move data to device after reshaping
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the Transformer model
    num_classes = len(class_to_label)
    model = TransformerModel(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=64,  # Adjust embedding dimension as needed
        num_heads=8,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 50  # Adjust the number of epochs as needed
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate the model on the test set and visualize results
    evaluate_and_visualize_model(model, test_loader, criterion, X_test, y_test, class_to_label, num_images=5)
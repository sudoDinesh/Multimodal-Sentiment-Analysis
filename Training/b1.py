import torch
from torch.utils.data import Dataset, DataLoader
import os
import csv
from torch import nn, optim

class MVSAFeatureDataset(Dataset):
    def __init__(self, image_features_dir, text_features_dir, labels_file):
        self.image_features_dir = image_features_dir
        self.text_features_dir = text_features_dir
        self.labels = self.load_labels(labels_file)
        self.data_ids = list(self.labels.keys())

    def load_labels(self, labels_file):
        labels = {}
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                id = int(row[0])
                sentiment = row[2].strip('"')
                labels[id] = self.label_to_int(sentiment)
        return labels

    def label_to_int(self, label):
        label_dict = {"positive": 2, "neutral": 1, "negative": 0}
        return label_dict[label]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image_feature_path = os.path.join(self.image_features_dir, f"{data_id}.pt")
        text_feature_path = os.path.join(self.text_features_dir, f"{data_id}.pt")

        image_features = torch.load(image_feature_path).squeeze(0).mean(dim=[1, 2])  # Mean pooling
        text_features = torch.load(text_feature_path).squeeze(0)  # CLS token representation

        # Ensure fixed-size feature vectors
        if image_features.size(0) != 2048:
            raise ValueError(f"Image feature size mismatch: {image_features.size(0)}")
        
        if text_features.dim() == 1:  # Handle cases where text_features is 1D
            if text_features.size(0) != 768:
                raise ValueError(f"Text feature size mismatch: {text_features.size(0)}")
            text_features = text_features.unsqueeze(0)  # Add a dimension to make it 2D
        elif text_features.size(1) != 768:
            raise ValueError(f"Text feature size mismatch: {text_features.size(1)}")

        features = torch.cat((image_features, text_features.squeeze(0)), dim=0)
        label = self.labels[data_id]

        return features, label

# Set paths
image_features_dir = r'/Users/dinesh/College/final proj/attempt2/features/data'
text_features_dir = r'/Users/dinesh/College/final proj/attempt2/features/text'
labels_file = r'/Users/dinesh/College/final proj/attempt2/pre-processing/MVSA_Single/labels.csv'

# Create the dataset and data loader
dataset = MVSAFeatureDataset(image_features_dir, text_features_dir, labels_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):
    def __init__(self, region_dim, word_dim, hidden_dim):
        super(CrossModalAlignment, self).__init__()
        self.region_proj = nn.Linear(region_dim, hidden_dim)
        self.word_proj = nn.Linear(word_dim, hidden_dim)

    def forward(self, image_features, text_features):
        # Reshape image_features and text_features
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, 2048]
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, 768]

        print(f"Reshaped image_features shape: {image_features.shape}")
        print(f"Reshaped text_features shape: {text_features.shape}")

        image_proj = self.region_proj(image_features)  # [batch_size, 1, hidden_dim]
        text_proj = self.word_proj(text_features)  # [batch_size, 1, hidden_dim]

        print(f"image_proj shape: {image_proj.shape}")
        print(f"text_proj shape: {text_proj.shape}")

        affinity_matrix = torch.matmul(image_proj, text_proj.transpose(1, 2))  # [batch_size, 1, 1]
        affinity_matrix = F.softmax(affinity_matrix / (hidden_dim ** 0.5), dim=-1)

        interactive_text_features = torch.matmul(affinity_matrix, text_proj)  # [batch_size, 1, hidden_dim]
        return interactive_text_features, image_proj  # Return the projected image features

class CrossModalGating(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalGating, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_proj, interactive_text_features):
        gate_values = self.sigmoid(torch.sum(image_proj * interactive_text_features, dim=-1))
        fused_features = gate_values.unsqueeze(-1) * interactive_text_features + image_proj
        return fused_features

class ITIN(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, hidden_dim, num_classes):
        super(ITIN, self).__init__()
        self.cross_modal_alignment = CrossModalAlignment(image_feature_dim, text_feature_dim, hidden_dim)
        self.cross_modal_gating = CrossModalGating(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, image_features, text_features):
        interactive_text_features, image_proj = self.cross_modal_alignment(image_features, text_features)
        fused_features = self.cross_modal_gating(image_proj, interactive_text_features)
        context_features = torch.mean(fused_features, dim=1)  # Aggregate region features
        
        x = self.fc1(context_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set parameters
image_feature_dim = 2048  # Based on Faster R-CNN's ResNet-101 backbone output channels
text_feature_dim = 768  # Based on BERT-base output hidden state size
hidden_dim = 512
num_classes = 3  # Adjust based on your sentiment labels

# Initialize the model, loss function, and optimizer
model = ITIN(image_feature_dim, text_feature_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in dataloader:
        image_features = features[:, :image_feature_dim]
        text_features = features[:, image_feature_dim:]

        # Debug prints
        print(f"Batch image_features shape: {image_features.shape}")
        print(f"Batch text_features shape: {text_features.shape}")

        optimizer.zero_grad()
        outputs = model(image_features, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training completed.")

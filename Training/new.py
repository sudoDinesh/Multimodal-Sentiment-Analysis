import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.models import resnet101
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from transformers import BertModel, BertTokenizer
from PIL import Image
import os
import pandas as pd
import re
import string
from torch.utils.data import Dataset, DataLoader

# Define dataset paths
image_dir = '/Users/dinesh/College/final proj/attempt2/features/data'
text_dir = '/Users/dinesh/College/final proj/attempt2/features/text'
labels_path = '/Users/dinesh/College/final proj/attempt2/MVSA_Single/labels.csv'

# Load the CSV file
labels_df = pd.read_csv(labels_path)

# Define the dataset class
class MultimodalDataset(Dataset):
    def __init__(self, labels_df, image_dir, text_dir):
        self.labels_df = labels_df
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row['id']
        sentiment = row['sentiment']

        # Load image features
        image_path = os.path.join(self.image_dir, f"{image_id}.pt")
        image_features = torch.load(image_path).squeeze(0)

        # Pad image features to a fixed size
        padded_image_features = torch.zeros((2048, 20, 20))
        c, h, w = image_features.size()
        padded_image_features[:, :h, :w] = image_features

        # Load text features
        text_path = os.path.join(self.text_dir, f"{image_id}.pt")
        text_features = torch.load(text_path)

        # Convert sentiment label to numerical
        sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        sentiment_label = sentiment_mapping[sentiment]

        return padded_image_features, text_features, sentiment_label

# Define the model components
class CrossModalAlignmentModule(nn.Module):
    def __init__(self, input_dim):
        super(CrossModalAlignmentModule, self).__init__()
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)

    def forward(self, image_features, text_features):
        # Perform cross-modal attention
        attn_output, _ = self.cross_modal_attention(image_features.flatten(2).permute(2, 0, 1), text_features, text_features)
        return attn_output

class CrossModalGatingModule(nn.Module):
    def __init__(self, input_dim):
        super(CrossModalGatingModule, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, image_features, text_features):
        gate_value = torch.sigmoid(self.gate(image_features))
        fused_features = gate_value * text_features + (1 - gate_value) * image_features
        return fused_features

class MultimodalSentimentAnalysisModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultimodalSentimentAnalysisModel, self).__init__()
        self.alignment_module = CrossModalAlignmentModule(input_dim)
        self.gating_module = CrossModalGatingModule(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, image_features, text_features):
        aligned_features = self.alignment_module(image_features, text_features)
        fused_features = self.gating_module(aligned_features, text_features)
        output = self.fc(fused_features.mean(dim=0))
        return output

# Define training parameters
input_dim = 2048  # Example input dimension, adjust as necessary
num_classes = 3   # Number of sentiment classes (positive, neutral, negative)
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Initialize dataset and dataloader
dataset = MultimodalDataset(labels_df, image_dir, text_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MultimodalSentimentAnalysisModel(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for image_features, text_features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(image_features, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), 'multimodal_sentiment_model.pth')

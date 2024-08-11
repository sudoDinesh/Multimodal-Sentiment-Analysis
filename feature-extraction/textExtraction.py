import torch
from transformers import BertModel, BertTokenizer
import os
import re
import string

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Define a function to load and preprocess the MVSA dataset text files
def load_text(text_path):
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
    except UnicodeDecodeError:
        with open(text_path, 'r', encoding='latin-1') as file:
            text = file.read().strip()
    return text

# Define a function to preprocess tweets
def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#'
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove special characters, numbers, etc.
    tweet = re.sub(r'\d+', '', tweet)
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

# Define a function to extract features from a text
def extract_text_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the CLS token representation
    features = outputs.last_hidden_state[:, 0, :]  # CLS token representation
    return features

# Directory containing the dataset
data_dir = r'/Users/dinesh/College/final proj/attempt2/MVSA_Single/data'

# Create a directory to save the extracted features
text_features_dir = '/Users/dinesh/College/final proj/attempt2/features/text'
os.makedirs(text_features_dir, exist_ok=True)

# Process all text files in the dataset
for i in range(1, 4869 + 1):
    text_path = os.path.join(data_dir, f"{i}.txt")
    if os.path.exists(text_path):
        text = load_text(text_path)
        preprocessed_text = preprocess_tweet(text)
        features = extract_text_features(preprocessed_text)
        # Save the extracted features
        feature_path = os.path.join(text_features_dir, f"{i}.pt")
        torch.save(features, feature_path)
        print(f"Extracted features for text {i}.txt and saved to {feature_path}")
    else:
        print(f"Text file {i}.txt not found.")

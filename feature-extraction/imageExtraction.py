
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.models import resnet101
from PIL import Image
import os

# Define a function to load and preprocess the MVSA dataset images
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)  # Convert PIL image to tensor
    return image

# Load a pre-trained ResNet-101 model
backbone = resnet101(pretrained=True)
# Remove the fully connected layer
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

# FasterRCNN needs to know the number of output channels in the backbone
backbone.out_channels = 2048

# Generate anchors using default anchor sizes and aspect ratios
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Define the RoI pooling layer
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=7, sampling_ratio=2
)

# Create the Faster R-CNN model using the custom backbone
model = FasterRCNN(
    backbone,
    num_classes=91,  # COCO has 91 classes
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

model.eval()  # Set the model to evaluation mode

# Define a function to extract features from an image
def extract_features(image_path):
    image = load_image(image_path)
    print(image_path)
    with torch.no_grad():
        # Get the feature maps from the backbone of the Faster R-CNN model
        features = model.backbone(image.unsqueeze(0))
    return features

# Directory containing the dataset
data_dir = r'/Users/dinesh/College/final proj/attempt2/MVSA_Single/data'

# Create a directory to save the extracted features
features_dir = '/Users/dinesh/College/final proj/attempt2/features/data'
os.makedirs(features_dir, exist_ok=True)

# Process all images in the dataset
for i in range(1, 4869 + 1):
    image_path = os.path.join(data_dir, f"{i}.jpg")
    if os.path.exists(image_path):
        features = extract_features(image_path)
        # Save the extracted features
        feature_path = os.path.join(features_dir, f"{i}.pt")
        torch.save(features, feature_path)
        print(f"Extracted features for image {i}.jpg and saved to {feature_path}")
    else:
        print(f"Image {i}.jpg not found.")

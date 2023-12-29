import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define a custom module that includes the reduction layer and image preprocessing
class CustomResNet50(nn.Module):
    def __init__(self, token_features_dim=1024):
        super(CustomResNet50, self).__init__()

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load a ResNet-50 model without pre-trained
        self.resnet50 = models.resnet50(weights=None)
        # Remove the classifier (fully connected) heasd
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        # Reduction layer
        self.reduction_layer = nn.Conv2d(2048, token_features_dim, kernel_size=2, stride=2)

    def forward(self, x):
        # Preprocess the image
        x = self.transform(x)

        # Extract features using the ResNet-50 model
        features = self.resnet50(x)

        # Apply the reduction layer
        features = self.reduction_layer(features)
        return features

if __name__ == '__main__':
    # Create an instance of the custom model
    custom_resnet50 = CustomResNet50()

    # Define a function to extract features from an image
    def extract_features(image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        features = custom_resnet50(image)

        return features

    # Example usage:
    image_path = 'path_to_your_image.jpg'
    features = extract_features(image_path)
    print(features.shape)  # Print the shape of the extracted features

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerModel, self).__init__()
        # Use ResNet50 as the base model
        self.base_model = models.resnet50(pretrained=True)
        # Replace the last layer for our classification task
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Weight the channels by corresponding gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        # Take the weighted average
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Pass through ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def visualize_gradcam(image_path, model, target_layer, class_names):
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    cam = grad_cam.generate_cam(input_tensor)
    
    # Load original image
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (224, 224))
    
    # Convert heatmap to RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    result = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_class = class_names[pred.item()]
        
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {pred_class}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

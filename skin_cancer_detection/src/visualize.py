import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

from torchvision import models

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(SkinCancerModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in self.resnet.parameters():
            param.requires_grad = False

        if fine_tune:
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Function to load class names from the training directory
def get_class_names(data_dir='../data/processed/train'):
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        return None
    return sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])

# A robust Grad-CAM implementation using modern hooks
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.model.eval()
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def _get_cam_weights(self, grads):
        return torch.mean(grads, dim=[2, 3], keepdim=True)

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Check hook registration.")

        cam_weights = self._get_cam_weights(self.gradients)
        cam = torch.sum(cam_weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        return cam.detach(), class_idx, output.detach()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def visualize_and_save(image_path, cam_tensor, predicted_class_name, confidence):
    original_img = cv2.imread(image_path)
    heatmap = cam_tensor.cpu().numpy()
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # Create a comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Convert images from OpenCV BGR to Matplotlib RGB for correct color display
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Display original image
    ax1.imshow(original_img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Display Grad-CAM image
    ax2.imshow(superimposed_img_rgb)
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')

    # Add a main title with detailed information
    true_class = os.path.basename(os.path.dirname(image_path))
    fig.suptitle(f'Predicted: {predicted_class_name} ({confidence:.2f}%)\nTrue Class: {true_class}', fontsize=16)

    # Save the final plot
    output_path = os.path.join('../', 'gradcam_visualization.png')
    plt.savefig(output_path)
    print(f"Comparison visualization saved to: {output_path}")
    print(f"Predicted class: {predicted_class_name} with {confidence:.2%} confidence.")

def main():
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for a skin cancer model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_path', type=str, default='../models/best_skin_cancer_model.pth', help='Path to the trained model.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = get_class_names()
    if not class_names:
        return
    num_classes = len(class_names)

    model = SkinCancerModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Unfreeze the target layer for gradient calculation
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True
    target_layer = model.resnet.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    print(f"Generating Grad-CAM for: {args.image_path}")
    cam, predicted_idx, output = grad_cam.generate_cam(input_tensor)
    grad_cam.remove_hooks()

    # Get confidence score
    probabilities = torch.softmax(output, dim=1)[0]
    confidence = probabilities[predicted_idx].item()

    visualize_and_save(args.image_path, cam, class_names[predicted_idx], confidence)

if __name__ == "__main__":
    main()

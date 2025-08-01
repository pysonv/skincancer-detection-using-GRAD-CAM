import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate_model(model_path, test_dir, device):
    test_transform = get_test_transform()
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Found {num_classes} classes: {class_names}")

    model = SkinCancerModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('../confusion_matrix.png')
    print("\nConfusion matrix saved to ../confusion_matrix.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_path = '../models/best_skin_cancer_model.pth'
    test_dir = '../data/processed/test'

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    evaluate_model(model_path, test_dir, device)

if __name__ == "__main__":
    main()

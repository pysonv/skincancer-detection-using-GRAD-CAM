import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skin_cancer_model import SkinCancerModel
from train import get_transforms

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = '../data/processed'
    test_dir = os.path.join(data_dir, 'test')
    model_path = '../models/best_skin_cancer_model.pth'

    _, val_transform = get_transforms()
    test_dataset = ImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    model = SkinCancerModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    accuracy, cm = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    plot_confusion_matrix(cm, class_names, '../models/confusion_matrix.png')

if __name__ == "__main__":
    main()

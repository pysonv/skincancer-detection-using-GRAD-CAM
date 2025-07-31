import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from skin_cancer_model import SkinCancerModel
from train import get_transforms

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    output_path = '../models/confusion_matrix.png'
    plt.savefig(output_path)
    print(f'Confusion matrix saved to {output_path}')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load test data
    data_dir = '../data/processed'
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(test_dir):
        print(f"Test data not found at {test_dir}. Please run data_preprocessing.py first.")
        return

    _, test_transform = get_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Load model
    model_path = '../models/best_skin_cancer_model.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train.py first.")
        return

    model = SkinCancerModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    evaluate_model(model, test_loader, device, class_names)

if __name__ == '__main__':
    main()

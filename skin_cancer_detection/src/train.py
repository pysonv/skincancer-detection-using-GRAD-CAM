import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def get_transforms():
    # Define the transformations for the training and validation sets
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs, device):
    
    # Calculate class weights to handle class imbalance
    targets = torch.tensor(train_loader.dataset.targets)
    class_counts = torch.bincount(targets).float()
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(train_loader.dataset.classes)
    class_weights = class_weights.to(device)
    print(f"Using class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Using a lower learning rate for fine-tuning
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # Add a progress bar for the training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            # Add a progress bar for the validation loop
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), '../models/best_skin_cancer_model.pth')
            print("New best model saved!")
        print("--------------------------------------------------")
        
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_transform, val_transform = get_transforms()
    
    data_dir = '../data/processed'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
        # Create the model with fine-tuning enabled
    model = SkinCancerModel(num_classes=num_classes, fine_tune=True).to(device)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=20, device=device
    )

if __name__ == "__main__":
    main()

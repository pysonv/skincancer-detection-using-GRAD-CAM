import torch
import torch.nn as nn
from torchvision import models

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinCancerModel, self).__init__()
        # Load a pre-trained ResNet50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all the parameters in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Get the number of input features for the classifier
        num_ftrs = self.base_model.fc.in_features
        
        # Replace the final fully connected layer with a new one
        # The new layer has a dropout layer followed by a linear layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

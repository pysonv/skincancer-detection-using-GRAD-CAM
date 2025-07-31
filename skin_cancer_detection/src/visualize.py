import torch
import argparse
import os
from skin_cancer_model import SkinCancerModel, visualize_gradcam

def main():
    parser = argparse.ArgumentParser(description='Visualize skin cancer predictions with Grad-CAM')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--model_path', type=str, 
                       default='../models/best_skin_cancer_model.pth',
                       help='Path to the trained model')

    
    args = parser.parse_args()
    
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} not found!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        print("Please train the model first using train.py")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Determine class names and number of classes from the data directory
    train_dir = '../data/processed/train'
    if not os.path.isdir(train_dir):
        print(f"Error: Training data directory not found at {train_dir}.")
        print("Please run data_preprocessing.py and train.py first.")
        return
    
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Load model
    model = SkinCancerModel(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get the target layer for Grad-CAM (last convolutional layer)
    target_layer = model.base_model.layer4[-1].conv3
    
    # Visualize with Grad-CAM
    print(f"Generating Grad-CAM visualization for: {args.image_path}")
    visualize_gradcam(args.image_path, model, target_layer, class_names)

if __name__ == "__main__":
    main()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_generator(train_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return train_generator

def get_validation_generator(val_dir, img_size, batch_size):
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return validation_generator

import pandas as pd
import numpy as np
from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split
import requests
import zipfile
from tqdm import tqdm

def download_isic_sample_data():
    """
    Download a sample of ISIC skin cancer data for demonstration.
    In practice, you would download the full ISIC dataset.
    """
    print("Note: This is a template for downloading ISIC data.")
    print("For the full dataset, visit: https://www.isic-archive.com/")
    print("You'll need to register and download the official dataset.")
    
    # Create data directories
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/processed/train/benign', exist_ok=True)
    os.makedirs('../data/processed/train/malignant', exist_ok=True)
    os.makedirs('../data/processed/val/benign', exist_ok=True)
    os.makedirs('../data/processed/val/malignant', exist_ok=True)
    os.makedirs('../data/processed/test/benign', exist_ok=True)
    os.makedirs('../data/processed/test/malignant', exist_ok=True)
    
    print("Data directories created successfully!")

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess images by resizing and normalizing.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Load and resize image
                img_path = os.path.join(input_dir, filename)
                image = Image.open(img_path).convert('RGB')
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save preprocessed image
                output_path = os.path.join(output_dir, filename)
                image.save(output_path, 'JPEG', quality=95)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def create_dataset_split(data_dir, test_size=0.2, val_size=0.2):
    """
    Split dataset into train, validation, and test sets.
    """
    # Get all image paths and labels
    image_paths = []
    labels = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_name)
    
    # Convert labels to numeric
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels]
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, numeric_labels, test_size=test_size, random_state=42, stratify=numeric_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'label_to_idx': label_to_idx,
        'idx_to_label': {idx: label for label, idx in label_to_idx.items()}
    }

def copy_split_data(split_data, output_base_dir):
    """
    Copy split data to organized directory structure.
    """
    for split_name, (paths, labels) in split_data.items():
        if split_name in ['train', 'val', 'test']:
            for path, label in zip(paths, labels):
                # Get class name from numeric label
                class_name = split_data['idx_to_label'][label]
                
                # Create destination directory
                dest_dir = os.path.join(output_base_dir, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy file
                filename = os.path.basename(path)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(path, dest_path)

def create_sample_dataset():
    """
    Create a sample dataset structure for demonstration.
    """
    print("Creating sample dataset structure...")
    
    # Create sample data directories
    sample_dir = '../data/sample'
    os.makedirs(f'{sample_dir}/benign', exist_ok=True)
    os.makedirs(f'{sample_dir}/malignant', exist_ok=True)
    
    # Create placeholder images (in practice, you'd have real images)
    from PIL import Image, ImageDraw
    import random
    
    # Generate sample benign images (lighter colors)
    for i in range(20):
        img = Image.new('RGB', (224, 224), color=(random.randint(150, 255), 
                                                  random.randint(150, 200), 
                                                  random.randint(100, 150)))
        draw = ImageDraw.Draw(img)
        # Add some random shapes to simulate skin lesions
        for _ in range(3):
            x1, y1 = random.randint(0, 150), random.randint(0, 150)
            x2, y2 = x1 + random.randint(20, 70), y1 + random.randint(20, 70)
            color = (random.randint(100, 180), random.randint(80, 120), random.randint(60, 100))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        img.save(f'{sample_dir}/benign/benign_{i:03d}.jpg')
    
    # Generate sample malignant images (darker, more irregular)
    for i in range(20):
        img = Image.new('RGB', (224, 224), color=(random.randint(120, 180), 
                                                  random.randint(100, 150), 
                                                  random.randint(80, 120)))
        draw = ImageDraw.Draw(img)
        # Add irregular shapes to simulate malignant lesions
        for _ in range(5):
            x1, y1 = random.randint(0, 150), random.randint(0, 150)
            x2, y2 = x1 + random.randint(30, 80), y1 + random.randint(30, 80)
            color = (random.randint(40, 100), random.randint(20, 80), random.randint(10, 60))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        img.save(f'{sample_dir}/malignant/malignant_{i:03d}.jpg')
    
    print(f"Sample dataset created with 40 images in {sample_dir}")
    return sample_dir

def process_multiclass_data(data_dir):
    print(f"Processing multi-class data from: {data_dir}")
    
    # Find class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not class_names:
        print(f"Error: No class subdirectories found in {data_dir}")
        return
    print(f"Found {len(class_names)} classes: {class_names}")

    # Collect all image paths and labels
    image_paths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

    if not image_paths:
        print("Error: No images found in the class subdirectories.")
        return

    print(f"Found a total of {len(image_paths)} images.")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    split_data = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    # Setup processed directories
    output_base_dir = '../data/processed'
    if os.path.exists(output_base_dir):
        print(f"Cleaning old processed data from {output_base_dir}")
        shutil.rmtree(output_base_dir)

    print(f"Copying split data to {output_base_dir}")
    for split_name, (paths, labels) in split_data.items():
        print(f"Processing {split_name} set...")
        for path, label in tqdm(zip(paths, labels), total=len(paths)):
            dest_dir = os.path.join(output_base_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(path, dest_dir)

    print("\nData preprocessing completed!")
    print("You can now run train.py to train the model on your new multi-class dataset.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess a multi-class skin cancer dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory (containing class sub-folders)')
    args = parser.parse_args()
    
    process_multiclass_data(args.data_dir)


if __name__ == "__main__":
    main()

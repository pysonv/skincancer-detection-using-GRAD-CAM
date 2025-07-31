# Multi-Class Skin Cancer Detection with Grad-CAM

This project implements a deep learning pipeline for multi-class skin cancer classification using a ResNet50 architecture and visualizes the model's predictions using Grad-CAM. The goal is to build an interpretable model that can distinguish between different types of skin lesions and highlight the image regions that influence its decisions.

## Project Status

The model has been adapted for a multi-class problem and trained on a small, custom dataset of 7 skin lesion types. The final model achieves an accuracy of **41.18%** on the test set. 

While techniques like data augmentation and dropout were implemented to combat overfitting, the model's performance is still heavily constrained by the limited size of the dataset. The project serves as a solid foundation, but significant performance gains would require a much larger and more balanced dataset.

## Features

- **Model**: A modified ResNet50 model for multi-class image classification.
- **Data Preprocessing**: Scripts to automatically split and organize image data from class-labeled folders.
- **Training**: A complete training pipeline with data augmentation, learning rate scheduling, and best-model saving.
- **Evaluation**: A script to measure test set accuracy and generate a confusion matrix to analyze class-wise performance.
- **Interpretability**: Grad-CAM visualization to produce heatmaps that show which parts of an image the model focuses on for its predictions.

## Project Structure

```
skin_cancer_detection/
├── data/
│   ├── raw/ (Original dataset location)
│   └── processed/ (Split data: train/val/test)
├── models/
│   ├── best_skin_cancer_model.pth
│   ├── training_history.png
│   └── confusion_matrix.png
├── src/
│   ├── skin_cancer_model.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

## How to Use

### 1. Setup

First, clone the repository and install the required dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your dataset into a directory where each sub-folder is named after a class and contains the corresponding images.

```
data/raw/my_dataset/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
│   ├── image2.jpg
│   └── ...
└── ...
```

### 3. Preprocess the Data

Run the preprocessing script to split your data into training, validation, and test sets.

```bash
cd src
python data_preprocessing.py --data_dir "../data/raw/my_dataset"
```

### 4. Train the Model

Train the model on the preprocessed data. The best model will be saved to `models/best_skin_cancer_model.pth`.

```bash
python train.py
```

### 5. Evaluate the Model

Evaluate the trained model on the test set. This will print the accuracy and save a confusion matrix plot.

```bash
python evaluate.py
```

### 6. Visualize Predictions

Generate a Grad-CAM visualization for a specific image to understand the model's prediction.

```bash
python visualize.py --image_path "/path/to/your/image.jpg"
```


This project implements a skin cancer detection system using deep learning and Grad-CAM for visualization.

## Features

- Skin cancer classification using a modified ResNet50 model
- Grad-CAM visualization for model interpretability
- Pre-trained on ISIC dataset (recommended)
- Easy-to-use visualization interface

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download and preprocess the ISIC skin cancer dataset
2. Train the model:
   ```
   python train.py
   ```
3. Visualize predictions with Grad-CAM:
   ```
   python visualize.py --image_path <path_to_image>
   ```

## Project Structure

- `src/`: Source code
  - `skin_cancer_model.py`: Main model implementation and Grad-CAM
- `data/`: Dataset storage
- `models/`: Trained model checkpoints

## Model Architecture

The model uses a modified ResNet50 architecture with:
- Pre-trained ResNet50 as base
- Custom classification head
- Grad-CAM implementation for visualization

## Visualization

The Grad-CAM visualization helps interpret the model's predictions by highlighting important regions in the input image.

## License

MIT License

import os
import shutil
import glob
from sklearn.model_selection import train_test_split

# --- Configuration ---
# The source directory containing the raw ISIC dataset, with subfolders for each class.
SOURCE_BASE_DIR = 'C:\\Users\\Pyson v\\Downloads\\archive\\Skin cancer ISIC The International Skin Imaging Collaboration'
# The destination directory where the processed (train/val/test) data will be saved.
PROCESSED_BASE_DIR = os.path.join('..', 'data', 'processed')
# The ratio for splitting data into training, validation, and testing sets.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
# A fixed random state for reproducibility of the data split.
RANDOM_STATE = 42

# --- Function Definitions ---

def clean_and_create_dirs():
    """Deletes the old processed directory and creates a fresh, empty structure."""
    if os.path.exists(PROCESSED_BASE_DIR):
        print(f"Removing existing processed directory: {PROCESSED_BASE_DIR}")
        shutil.rmtree(PROCESSED_BASE_DIR)
    
    print("Creating new directory structure...")
    os.makedirs(PROCESSED_BASE_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(PROCESSED_BASE_DIR, split), exist_ok=True)

def split_and_copy_files():
    """Discovers classes, splits files, and copies them to the correct destination."""
    # Find all class directories in the source training data folder.
    source_train_dir = os.path.join(SOURCE_BASE_DIR, 'Train')
    class_names = [d for d in os.listdir(source_train_dir) if os.path.isdir(os.path.join(source_train_dir, d))]
    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")
        
        # Create class subdirectories in train, val, and test.
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(PROCESSED_BASE_DIR, split, class_name), exist_ok=True)

        # Get all image paths for the current class from both Train and Test source folders.
        source_class_dir_train = os.path.join(source_train_dir, class_name)
        source_class_dir_test = os.path.join(SOURCE_BASE_DIR, 'Test', class_name)
        
        all_images = []
        if os.path.exists(source_class_dir_train):
            all_images.extend(glob.glob(os.path.join(source_class_dir_train, '*.jpg')))
        if os.path.exists(source_class_dir_test):
            all_images.extend(glob.glob(os.path.join(source_class_dir_test, '*.jpg')))
        
        if not all_images:
            print(f"  Warning: No images found for class {class_name}. Skipping.")
            continue

        # First split: separate out the training set.
        train_files, remaining_files = train_test_split(
            all_images, 
            train_size=TRAIN_RATIO, 
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # Second split: divide the remainder into validation and test sets.
        # The ratio needs to be recalculated for the remaining data.
        val_ratio_adjusted = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            remaining_files, 
            train_size=val_ratio_adjusted, 
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # Function to copy files to their destination.
        def copy_files(files, split_name):
            for file_path in files:
                shutil.copy(file_path, os.path.join(PROCESSED_BASE_DIR, split_name, class_name))
            print(f"  Copied {len(files)} files to {split_name}/{class_name}")

        # Copy the files to their final destinations.
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Data Preprocessing ---")
    clean_and_create_dirs()
    split_and_copy_files()
    print("\n--- Data Preprocessing Complete! ---")

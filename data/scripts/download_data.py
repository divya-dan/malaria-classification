import os
import zipfile
import shutil
from kaggle import api
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data')
)
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

# Kaggle dataset identifier (Cell Images for Detecting Malaria)
DATASET = 'iarunava/cell-images-for-detecting-malaria'
DOWNLOAD_PATH = RAW_DIR

# Create necessary directories
def create_dirs():
    for d in [RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

# Download and unzip dataset
def download_and_extract():
    # Check if data already exists (look for 'Parasitized' and 'Uninfected' folders)
    expected_dirs = [
        os.path.join(RAW_DIR, 'cell_images', 'Parasitized'),
        os.path.join(RAW_DIR, 'cell_images', 'Uninfected'),
        os.path.join(RAW_DIR, 'cell_images', 'cell_images', 'Parasitized'),
        os.path.join(RAW_DIR, 'cell_images', 'cell_images', 'Uninfected'),
        os.path.join(RAW_DIR, 'Parasitized'),
        os.path.join(RAW_DIR, 'Uninfected'),
    ]
    if all(os.path.isdir(os.path.dirname(d)) and os.path.isdir(d) for d in expected_dirs[:2]) or \
       all(os.path.isdir(os.path.dirname(d)) and os.path.isdir(d) for d in expected_dirs[2:4]) or \
       all(os.path.isdir(d) for d in expected_dirs[4:]):
        print("Data already exists, skipping download.")
        return

    print(f"Downloading dataset {DATASET} into {DOWNLOAD_PATH}...")
    api.dataset_download_files(DATASET, path=DOWNLOAD_PATH, unzip=False)
    zip_name = DATASET.split('/')[-1] + '.zip'
    zip_path = os.path.join(DOWNLOAD_PATH, zip_name)
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    os.remove(zip_path)

# Split data into train/val/test
def split_data(test_size=0.15, val_size=0.15, random_state=42):
    # The extracted folder may have nested 'cell_images/cell_images'
    # Find the correct base directory containing 'Parasitized' and 'Uninfected'
    possible_dirs = [
        os.path.join(RAW_DIR, 'cell_images', 'cell_images'),
        os.path.join(RAW_DIR, 'cell_images'),
        RAW_DIR
    ]
    base_dir = None
    for d in possible_dirs:
        if all(os.path.isdir(os.path.join(d, cls)) for cls in ['Parasitized', 'Uninfected']):
            base_dir = d
            break
    if base_dir is None:
        raise RuntimeError("Could not find 'Parasitized' and 'Uninfected' folders in extracted data.")

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Stratified splits
        train_val, test = train_test_split(
            images, test_size=test_size,
            stratify=[cls] * len(images), random_state=random_state
        )
        val_relative_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_relative_size,
            stratify=[cls] * len(train_val), random_state=random_state
        )

        # Helper to copy files
        print(f"Copying {len(train)} {cls} images to train...")
        copy_files(train, TRAIN_DIR, cls)
        print(f"Copying {len(val)} {cls} images to val...")
        copy_files(val, VAL_DIR, cls)
        print(f"Copying {len(test)} {cls} images to test...")
        copy_files(test, TEST_DIR, cls)

def copy_files(file_list, dest_dir, cls):
    target_dir = os.path.join(dest_dir, cls)
    os.makedirs(target_dir, exist_ok=True)
    for src in file_list:
        shutil.copy(src, target_dir)

if __name__ == '__main__':
    create_dirs()
    download_and_extract()
    split_data()
    print("Data download and split complete.")


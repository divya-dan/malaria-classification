import os
import torch

# Project root (one level up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Ensure output dirs exist
for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Training hyperparameters
SEED = 19
BATCH_SIZE = 256
NUM_WORKERS = 8
IMG_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
AUGMENT = True  # Toggle augmentation 

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
IN_CHANNELS = 3
NUM_CLASSES = 1  # binary

# Save checkpoint naming
BEST_MODEL_NAME = f"simple_cnn_best{'_aug' if AUGMENT else ''}.pth"

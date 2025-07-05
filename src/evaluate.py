import os
import torch
import numpy as np
from config import DEVICE, MODEL_DIR, BEST_MODEL_NAME, AUGMENT
from utils import compute_metrics, plot_confusion_matrix, plot_training_curves
from datamodule import MalariaDataModule
from model import SimpleCNN

def evaluate():
    # Load DataModule and test loader
    dm = MalariaDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    # Load best model checkpoint
    model = SimpleCNN().to(DEVICE)
    checkpoint_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Run inference
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.cpu().numpy()

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    # Compute and print metrics
    metrics = compute_metrics(y_true, y_probs)
    print("Test set performance:")
    for k, v in metrics.items():
        print(f"  {k.title()}: {v:.4f}")

    # Plot confusion matrix
    classes = dm.classes
    cm_path = os.path.join(MODEL_DIR, f'confusion_matrix{"_aug" if AUGMENT else ""}.png')
    plot_confusion_matrix(y_true, y_pred, classes, normalize=True, save_fig=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Plot training curves if available
    history_path = os.path.join(MODEL_DIR, f'training_history{"_aug" if AUGMENT else ""}.pt')
    if os.path.exists(history_path):
        history = torch.load(history_path)
        plot_training_curves(history)
        print("Training curves saved to outputs/figures")
    else:
        print(f"Training history not found at {history_path}, skipping curves.")

if __name__ == '__main__':
    evaluate()

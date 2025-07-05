import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MODEL_DIR, BEST_MODEL_NAME, LOG_DIR
)
from utils import set_seed, compute_metrics
from datamodule import MalariaDataModule
from model import SimpleCNN
from tqdm import tqdm


def train():
    # Set random seed
    set_seed()

    # Data
    dm = MalariaDataModule()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Model, loss, optimizer
    model = SimpleCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # TensorBoard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, timestamp))

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
    }

    best_val_loss = float('inf')
    best_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Collect for accuracy
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_probs=np.array(all_preds)
        )
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(metrics['accuracy'])

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_metrics(
            y_true=np.array(val_labels),
            y_probs=np.array(val_preds)
        )
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f}, Acc: {metrics['accuracy']:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    writer.close()

    # Save final history
    history_path = os.path.join(MODEL_DIR, 'training_history.pt')
    torch.save(history, history_path)
    print(f"Training complete. History saved to {history_path}")


if __name__ == '__main__':
    train()

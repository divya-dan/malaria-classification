import torch
import torch.nn as nn
from config import IN_CHANNELS, NUM_CLASSES, IMG_SIZE

class SimpleCNN(nn.Module):
    """
    A simple CNN for binary classification of malaria cell images.
    Architecture: 3 convolutional blocks followed by 2 fully-connected layers.
    Uses BCEWithLogitsLoss, so final layer has 1 output.
    """
    def __init__(self, in_channels: int = IN_CHANNELS, num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE):
        super(SimpleCNN, self).__init__()
        # Convolutional blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Compute the size after convolutions (assuming square input)
        conv_output_size = img_size // (2 ** 3)  # each block halves the spatial dims
        flattened_dim = 128 * conv_output_size * conv_output_size

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Quick model sanity check
    model = SimpleCNN()
    x = torch.randn(4, IN_CHANNELS, IMG_SIZE, IMG_SIZE)  # batch of 4
    logits = model(x)
    print(f"Output shape: {logits.shape} (expected [4, {NUM_CLASSES}])")
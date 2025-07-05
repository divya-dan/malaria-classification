import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, AUGMENT, SEED
from utils import set_seed

class MalariaDataModule:
    """
    DataModule for Malaria Cell Images using paths and params from config.
    Expects processed data under DATA_DIR/{train,val,test}/{Parasitized,Uninfected}.
    """
    def __init__(self):
        # Seed for reproducibility
        set_seed(SEED)

        # Paths
        self.train_dir = os.path.join(DATA_DIR, 'train')
        self.val_dir   = os.path.join(DATA_DIR, 'val')
        self.test_dir  = os.path.join(DATA_DIR, 'test')

        # Hyperparameters
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.img_size = IMG_SIZE
        self.augment = AUGMENT

        # Setup transforms
        self._init_transforms()

    def _init_transforms(self):
        # Base deterministic transforms
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ]
        self.test_transform = transforms.Compose(base_transforms)

        if self.augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.8,1.2)),
            ]
            self.train_transform = transforms.Compose(aug_transforms + base_transforms)
        else:
            self.train_transform = self.test_transform

    def setup(self):
        # Create datasets
        self.train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        self.val_dataset   = datasets.ImageFolder(root=self.val_dir,   transform=self.test_transform)
        self.test_dataset  = datasets.ImageFolder(root=self.test_dir,  transform=self.test_transform)

        # Metadata
        self.classes = self.train_dataset.classes
        self.num_classes = len(self.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

if __name__ == '__main__':
    dm = MalariaDataModule()
    dm.setup()
    print(f"Classes: {dm.classes}")
    batch = next(iter(dm.train_dataloader()))
    print(f"Train batch - images: {batch[0].shape}, labels: {batch[1].shape}")
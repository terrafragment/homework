import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(img_path).convert('RGB')
        label = int(self.image_files[index].split('_')[0])
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDataModule(LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.train_dir, transform=self.transform)
            self.val_dataset = CustomDataset(self.val_dir, transform=self.transform)
        elif stage == 'validate' or stage == 'test':
            self.val_dataset = CustomDataset(self.val_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == '__main__':
    # Debugging code to check the data loader
    train_dir = 'train_path'
    val_dir = 'val_path'
    data_module = CustomDataModule(train_dir, val_dir)

    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        print(type(batch), type(batch[0]), type(batch[1]))
        break

from pytorch_lightning import Trainer
from model import ViolenceClassifier
from dataset import CustomDataModule
import torch

if __name__ == '__main__':
    # Directory for validation data
    val_dir = 'val_path'
    batch_size = 128

    # Initialize the data module
    data_module = CustomDataModule(train_dir=None, val_dir=val_dir, batch_size=batch_size)

    # Load the trained model from checkpoint
    checkpoint_path = 'lightning_logs/version_4/checkpoints/epoch=10-val_acc=0.99.ckpt'
    model = ViolenceClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)

    # Set up the trainer with appropriate device configuration
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)

    # Set up the data module for the test stage
    data_module.setup(stage='test')

    # Run the test
    trainer.test(model, data_module.val_dataloader())

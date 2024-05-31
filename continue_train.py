import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import CustomDataModule
from model import ViolenceClassifier

def main():
    train_dir = 'train_path'
    val_dir = 'val_path'
    batch_size = 32
    num_classes = 2
    pretrained_model_path = 'checkpoint_path'  # 已有模型的 checkpoint 文件路径

    data_module = CustomDataModule(train_dir, val_dir, batch_size)
    model = ViolenceClassifier(num_classes, pretrained_model_path=pretrained_model_path)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='{epoch:02d}-{val_acc:.2f}'
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=40,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()

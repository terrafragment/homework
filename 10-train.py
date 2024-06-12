if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from model import ViolenceClassifier
    from dataset import CustomDataModule

    train_dir = 'train_path'
    val_dir = 'val_path'
    gpu_id = [0]
    lr = 3e-4
    batch_size = 128
    log_name = "resnet18_pretrain_test"

    print(f"{log_name} gpu: {gpu_id}, batch size: {batch_size}, lr: {lr}")

    # Initialize Data Module
    data_module = CustomDataModule(train_dir, val_dir, batch_size=batch_size)

    # Model checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # TensorBoard logger
    logger = TensorBoardLogger("train_logs", name=log_name)

    # Trainer setup
    trainer = Trainer(
        max_epochs=40,
        accelerator='gpu',
        devices=gpu_id,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Initialize model
    model = ViolenceClassifier(num_classes=2)

    # Setup data module
    data_module.setup(stage='fit')

    # Start training
    trainer.fit(model, data_module)

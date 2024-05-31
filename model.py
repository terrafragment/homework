import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class ViolenceClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained_model_path=None):
        super(ViolenceClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.test_outputs = []

        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)

    def load_pretrained_weights(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        model_state_dict = checkpoint['state_dict']

        # 过滤掉不匹配的键
        filtered_model_state_dict = {k: v for k, v in model_state_dict.items() if k in self.state_dict()}
        self.load_state_dict(filtered_model_state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data).double() / len(labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data).double() / len(labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.test_outputs.append({'test_loss': loss, 'test_acc': acc})
        return {'test_loss': loss, 'test_acc': acc}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['test_loss'] for x in self.test_outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in self.test_outputs]).mean()
        self.log('avg_test_loss', avg_loss)
        self.log('avg_test_acc', avg_acc)
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

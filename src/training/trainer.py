import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, auroc

from src.config import train_config


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=1)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05, num_samples: int = 0):
        super().__init__()
        self.num_samples = num_samples
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return torch.squeeze(out, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)

        wandb.log({"loss/train": loss})

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = F.sigmoid(logits)
        preds = probs > 0.5
        acc = accuracy(preds, y.int(), task="binary")
        auc = auroc(probs, y.int(), task="binary")
        # self.logger.experiment.log({"accuracy/val": acc, "auc/val": auc, "loss/val": loss})
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_auc", auc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = self.num_samples // train_config.BATCH_SIZE * 2
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

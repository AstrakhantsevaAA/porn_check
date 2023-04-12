from collections import defaultdict

import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import system_config, train_config
from src.data_utils.dataset import CustomDataset
from src.training.trainer import LitResnet

seed_everything(7)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    wandb.login()

    df = pd.read_csv(
        "/home/alenaastrakhantseva/PycharmProjects/porn_check/data/processed/splits/6_04_23.csv"
    )

    dataloaders = defaultdict()

    for split_type in ["train", "val", "test"]:
        dataset = CustomDataset(
            data_dir="/home/alenaastrakhantseva/PycharmProjects/porn_check",
            split_dataframe=df,
            split_type=split_type,
            augmentations_intensity=0.5,
        )

        sampler, shuffle = None, False
        if split_type == "train":
            weights = [1 / dataset.len_zeros, 1 / dataset.len_ones]
            samples_weights = [weights[int(label)] for label in dataset.labels]
            sampler = WeightedRandomSampler(
                weights=samples_weights, num_samples=len(dataset), replacement=True
            )

        dataloaders[split_type] = DataLoader(
            dataset,
            batch_size=train_config.BATCH_SIZE,
            num_workers=train_config.NUM_WORKERS,
            sampler=sampler,
        )

    model = LitResnet(lr=0.05, num_samples=len(dataloaders["train"].dataset))
    logger = WandbLogger(project="porn_check", save_dir=system_config.log_dir)

    trainer = Trainer(
        max_epochs=100,
        default_root_dir=system_config.model_dir,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    trainer.test(model, dataloaders["test"])

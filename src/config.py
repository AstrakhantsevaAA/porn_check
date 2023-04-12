import os
from pathlib import Path

import torch


class SystemConfig:
    root_dir: Path = Path(__file__).parents[1]
    model_dir: Path = root_dir / "models"
    data_dir: Path = root_dir / "data"
    log_dir: Path = root_dir / "logs"


class TrainConfig:
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8
    NUM_WORKERS = int(os.cpu_count() / 2)


train_config = TrainConfig()
system_config = SystemConfig()

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_utils.transforms import define_augmentations, define_transform
from src.config import system_config


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = image / 255.0
    return image.astype(np.float32)


def read_image(filepath: str) -> np.ndarray:
    image = cv2.imread(filepath)
    if image is None:
        raise Exception(f"image is None, got filepath: {filepath} \n data.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    return image


class CustomDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image tensors, and labels.
    """

    def __init__(
        self,
        data_dir: str,
        split_dataframe: pd.DataFrame,
        split_type: str = "train",
        augmentations_intensity: float = 0,
    ):
        self.data_dir = data_dir
        self.split_dataframe = split_dataframe
        self.data = self.split_dataframe[self.split_dataframe["split"] == split_type]
        self.labels = self.data.loc[:, "label"].values
        self.len_ones = sum(self.labels)
        self.len_zeros = len(self.labels) - self.len_ones

        self.transform = define_transform()
        self.augmentation = None
        if augmentations_intensity > 0:
            self.augmentation = define_augmentations(augmentations_intensity)

    def __getitem__(self, index: int):
        row = self.data.iloc[index, :]
        filepath = self.data_dir + "/" + str(row["filepath"])
        image = read_image(filepath)

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        image = self.transform(image=image)["image"]

        label = int(row["label"])
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    df = pd.read_csv(
        system_config.data_dir / "processed/splits/6_04_23.csv"
    )
    dataset = CustomDataset(
        data_dir=system_config.root_dir,
        split_dataframe=df,
        split_type="train",
        augmentations_intensity=0.8,
    )
    print(dataset.len_ones, dataset.len_zeros)

from albumentations import (
    Affine,
    Blur,
    CoarseDropout,
    ColorJitter,
    Compose,
    Downscale,
    Flip,
    GridDistortion,
    Perspective,
    RandomBrightnessContrast,
    RandomCrop,
    Resize,
    ShiftScaleRotate,
)
from albumentations.pytorch.transforms import ToTensorV2


def define_augmentations(augmentations_intensity: float = 0.0) -> Compose:
    return Compose(
        [
            CoarseDropout(
                p=0.5,
            ),
            Blur(p=0.5),
            ShiftScaleRotate(p=0.5),
            Affine(),
            GridDistortion(),
            Downscale(
                p=0.5,
            ),
            Perspective(p=0.5),
            Flip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            ColorJitter(),
        ],
        p=augmentations_intensity,
    )


def define_transform() -> Compose:
    return Compose(
        [
            Resize(256, 256),
            RandomCrop(224, 224),
            ToTensorV2(),
        ]
    )

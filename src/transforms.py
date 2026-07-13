import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from src.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool):
    aug = [A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH)]
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GridDistortion(distort_limit=0.15, p=0.2),
        ]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)

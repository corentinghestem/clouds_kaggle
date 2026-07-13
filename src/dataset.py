import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.rle import rle_decode


class CloudDataset(Dataset):
    def __init__(self, df, image_dir, transform, orig_shape, classes):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.orig_shape = orig_shape
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def _build_mask(self, row) -> np.ndarray:
        h, w = self.orig_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for idx, cls in enumerate(self.classes, start=1):
            rle = row.get(cls)
            if isinstance(rle, str) and rle.strip():
                mask[rle_decode(rle, (h, w)) == 1] = idx
        return mask

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.image_dir, row["image"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._build_mask(row)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].long()
        return image, mask

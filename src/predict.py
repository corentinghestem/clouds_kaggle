import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import Config
from src.model import build_model
from src.rle import rle_encode
from src.transforms import get_transforms


def predict_test_set():
    model = build_model()
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()

    transform = get_transforms(train=False)
    test_dir = os.path.join(Config.DATA_DIR, "test_images")
    image_names = sorted(os.listdir(test_dir))

    rows = []
    with torch.no_grad():
        for name in tqdm(image_names):
            image = cv2.cvtColor(cv2.imread(os.path.join(test_dir, name)), cv2.COLOR_BGR2RGB)
            dummy = np.zeros(image.shape[:2], dtype=np.uint8)
            tensor = transform(image=image, mask=dummy)["image"].unsqueeze(0).to(Config.DEVICE)

            probs = torch.softmax(model(pixel_values=tensor), dim=1)
            if Config.TTA_VAL:
                flipped = torch.flip(tensor, dims=[3])
                probs_f = torch.flip(torch.softmax(model(pixel_values=flipped), dim=1), dims=[3])
                probs = (probs + probs_f) / 2

            pred = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pred = cv2.resize(pred, (Config.ORIG_SHAPE[1], Config.ORIG_SHAPE[0]),
                              interpolation=cv2.INTER_NEAREST)

            for idx, cls in enumerate(Config.CLASSES, start=1):
                rows.append((f"{name}_{cls}", rle_encode(pred == idx)))

    sub = pd.DataFrame(rows, columns=["Image_Label", "EncodedPixels"])
    out_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Submission écrite : {out_path}")


if __name__ == "__main__":
    predict_test_set()

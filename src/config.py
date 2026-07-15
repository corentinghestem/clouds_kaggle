import os

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Config:
    IS_KAGGLE = os.path.isdir("/kaggle/input")
    if IS_KAGGLE:
        _CANDIDATES = [
            "/kaggle/input/competitions/understanding_cloud_organization",
            "/kaggle/input/understanding-cloud-organization",
        ]
        DATA_DIR = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])
        OUTPUT_DIR = "/kaggle/working/swin_unet_clouds"
    else:
        DATA_DIR = "./data"
        OUTPUT_DIR = "./swin_unet_clouds"

    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")

    BACKBONE_NAME = "microsoft/swin-base-patch4-window7-224-in22k"
    PRETRAINED_BACKBONE = True

    CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]
    NUM_CLASSES = len(CLASSES) + 1

    ORIG_SHAPE = (1400, 2100)
    IMG_HEIGHT = 512
    IMG_WIDTH = 768

    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 4
    NUM_EPOCHS = 35
    LR = 3e-5
    WEIGHT_DECAY = 1e-2
    WARMUP_RATIO = 0.05
    DICE_WEIGHT = 1.0
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 12
    VAL_SPLIT = 0.15
    NUM_WORKERS = 4 if IS_KAGGLE else 2
    SEED = 42
    USE_AMP = True
    TTA_VAL = True
    RESUME = True

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")


ID2LABEL = {0: "background", 1: "Fish", 2: "Flower", 3: "Gravel", 4: "Sugar"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

PALETTE = np.array([
    [0,   0,   0],
    [220, 20,  60],
    [255, 215, 0],
    [0,   191, 255],
    [50,  205, 50],
], dtype=np.uint8)

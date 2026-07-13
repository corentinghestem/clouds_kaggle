import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import Config, PALETTE
from src.transforms import get_transforms


@torch.no_grad()
def visualize_prediction(model, image_path, save_path):
    model.eval()
    transform = get_transforms(train=False)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    dummy = np.zeros(image.shape[:2], dtype=np.uint8)
    tensor = transform(image=image, mask=dummy)["image"].unsqueeze(0).to(Config.DEVICE)

    logits = model(pixel_values=tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    color_mask = PALETTE[pred]
    resized_img = cv2.resize(image, (Config.IMG_WIDTH, Config.IMG_HEIGHT))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(resized_img);   ax[0].set_title("Image satellite"); ax[0].axis("off")
    ax[1].imshow(color_mask);    ax[1].set_title("Masque prédit");   ax[1].axis("off")
    ax[2].imshow(resized_img);   ax[2].imshow(color_mask, alpha=0.45)
    ax[2].set_title("Superposition"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualisation enregistrée : {save_path}")

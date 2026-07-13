import csv
import math
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.config import Config, ID2LABEL
from src.dataset import CloudDataset
from src.engine import train_one_epoch, validate
from src.model import build_model
from src.transforms import get_transforms
from src.utils import set_seed, stratified_split
from src.visualize import visualize_prediction


def main():
    set_seed(Config.SEED)
    if Config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Device : {Config.DEVICE} | Kaggle : {Config.IS_KAGGLE}")

    df = pd.read_csv(Config.TRAIN_CSV)
    df[["image", "label"]] = df["Image_Label"].str.rsplit("_", n=1, expand=True)
    pivot = df.pivot(index="image", columns="label",
                     values="EncodedPixels").reset_index()
    print(f"Nombre d'images : {len(pivot)}")

    n_labels = pivot[Config.CLASSES].notna().sum(axis=1)
    train_df, val_df = stratified_split(
        pivot, n_labels, val_frac=Config.VAL_SPLIT, seed=Config.SEED)
    print(f"Train : {len(train_df)} images | Validation : {len(val_df)} images")

    train_ds = CloudDataset(train_df, Config.TRAIN_IMAGES_DIR,
                            get_transforms(True), Config.ORIG_SHAPE, Config.CLASSES)
    val_ds = CloudDataset(val_df, Config.TRAIN_IMAGES_DIR,
                          get_transforms(False), Config.ORIG_SHAPE, Config.CLASSES)

    pin = Config.DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=pin,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=pin)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR,
                                  weight_decay=Config.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / Config.GRAD_ACCUM_STEPS)
    total_steps = Config.NUM_EPOCHS * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )
    amp_enabled = Config.USE_AMP and Config.DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pt")
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "checkpoint_last.pt")
    start_epoch = 1
    best_miou = 0.0
    epochs_no_improve = 0
    if Config.RESUME and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        print(f"Reprise depuis l'époque {start_epoch} (meilleure mIoU = {best_miou:.4f})")

    log_path = os.path.join(Config.OUTPUT_DIR, "training_log.csv")
    write_header = not os.path.exists(log_path)

    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        print(f"\nÉpoque {epoch}/{Config.NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     scaler, Config.DEVICE, Config.GRAD_ACCUM_STEPS)
        val_loss, metric = validate(model, val_loader, Config.DEVICE,
                                    Config.NUM_CLASSES)
        miou = metric.mean_iou()

        print(f"  train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f} "
              f"| mIoU = {miou:.4f}")
        for name, iou in zip(ID2LABEL.values(), metric.iou_per_class()):
            print(f"     IoU {name:<11}: {iou:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "mIoU"]
                                + [f"IoU_{n}" for n in ID2LABEL.values()])
                write_header = False
            writer.writerow([epoch, train_loss, val_loss, miou]
                            + list(metric.iou_per_class()))

        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Meilleur modèle sauvegardé (mIoU = {best_miou:.4f})")
        else:
            epochs_no_improve += 1

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "epochs_no_improve": epochs_no_improve,
        }, ckpt_path)

        if epochs_no_improve >= Config.EARLY_STOP_PATIENCE:
            print(f"\nArrêt anticipé : pas d'amélioration depuis "
                  f"{Config.EARLY_STOP_PATIENCE} époques.")
            break

    print(f"\nEntraînement terminé. Meilleure mIoU : {best_miou:.4f}")

    best_model = build_model()
    best_model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    sample_image = os.path.join(Config.TRAIN_IMAGES_DIR, val_df.iloc[0]["image"])
    if os.path.exists(sample_image):
        visualize_prediction(
            best_model, sample_image,
            os.path.join(Config.OUTPUT_DIR, "exemple_prediction.png"))

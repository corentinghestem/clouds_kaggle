import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.losses import combined_loss
from src.metrics import SegmentationMetric


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps):
    model.train()
    amp_enabled = scaler.is_enabled()
    running = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  train", leave=False)
    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(pixel_values=images).float()
        loss = combined_loss(logits, masks, Config.NUM_CLASSES, Config.DICE_WEIGHT)

        scaler.scale(loss / accum_steps).backward()

        is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == n_batches
        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    metric = SegmentationMetric(num_classes)
    running = 0.0
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)

        logits = model(pixel_values=images)
        running += combined_loss(logits, masks, num_classes, Config.DICE_WEIGHT).item() * images.size(0)
        probs = F.softmax(logits, dim=1)

        if Config.TTA_VAL:
            flipped = torch.flip(images, dims=[3])
            logits_f = model(pixel_values=flipped)
            probs_f = torch.flip(F.softmax(logits_f, dim=1), dims=[3])
            probs = (probs + probs_f) / 2

        preds = probs.argmax(dim=1)
        metric.update(preds.cpu().numpy(), masks.cpu().numpy())
    return running / len(loader.dataset), metric

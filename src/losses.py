import torch
import torch.nn.functional as F


def dice_loss_foreground(logits: torch.Tensor, targets: torch.Tensor,
                         num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

    probs_fg = probs[:, 1:, :, :]
    targets_fg = targets_onehot[:, 1:, :, :]

    dims = (0, 2, 3)
    intersection = torch.sum(probs_fg * targets_fg, dims)
    cardinality = torch.sum(probs_fg + targets_fg, dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_per_class.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  num_classes: int, dice_weight: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets)
    dice = dice_loss_foreground(logits, targets, num_classes)
    return ce + dice_weight * dice

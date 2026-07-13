import numpy as np


class SegmentationMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        m = (targets >= 0) & (targets < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets[m].astype(int) + preds[m].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def iou_per_class(self) -> np.ndarray:
        c = self.confusion
        inter = np.diag(c)
        union = c.sum(1) + c.sum(0) - inter
        return inter / np.maximum(union, 1)

    def mean_iou(self) -> float:
        return float(np.mean(self.iou_per_class()))

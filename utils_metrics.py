# utils_metrics.py
from typing import Dict, List, Tuple
import numpy as np
import torch


def top1_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Dict[str, np.ndarray]:
    # cm[i,j]：真 i 被预测成 j
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / np.clip(tp + fp, 1e-12, None)
    recall = tp / np.clip(tp + fn, 1e-12, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    support = cm.sum(axis=1).astype(np.int64)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }

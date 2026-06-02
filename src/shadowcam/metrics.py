from __future__ import annotations

import numpy as np


def segmentation_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3):
    report = {}
    total = max(int(y_true.size), 1)
    for cls in range(num_classes):
        true = y_true == cls
        pred = y_pred == cls
        tp = np.logical_and(true, pred).sum()
        fp = np.logical_and(~true, pred).sum()
        fn = np.logical_and(true, ~pred).sum()
        union = tp + fp + fn
        report[f"class_{cls}_iou"] = float(tp / union) if union else 1.0
        report[f"class_{cls}_recall"] = float(tp / (tp + fn)) if (tp + fn) else 1.0
        report[f"class_{cls}_precision"] = float(tp / (tp + fp)) if (tp + fp) else 1.0
        report[f"class_{cls}_true_ratio"] = float(true.sum() / total)
        report[f"class_{cls}_pred_ratio"] = float(pred.sum() / total)
    report["hand_false_positive_rate"] = float(((y_true == 2) & (y_pred == 1)).sum() / max((y_true == 2).sum(), 1))
    report["dark_nonshadow_false_positive_rate"] = float(((y_true != 1) & (y_pred == 1)).sum() / max((y_true != 1).sum(), 1))
    return report

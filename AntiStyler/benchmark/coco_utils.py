"""Glue between COCO ground truth, torchvision's Faster R-CNN prediction
format, and pycocotools -- needed by evaluate.py for the mAP@0.5 scoring
(Table 1) and for the patch-validity filter from the paper's supplementary
material (Section 2.1.3): "we excluded images in which the patch only
caused partial occlusion without any adversarial effect by comparing the
predictions on the attacked image with and without a black mask on the
patched area." predictions_equivalent implements that comparison.

data.py's own comment already confirms COCO category ids (1-90, with gaps)
match torchvision Faster R-CNN's label ids directly -- ground truth labels
below are used as-is, no remapping.
"""
from __future__ import annotations

import numpy as np


def ground_truth_for_image(coco, image_id: int) -> dict:
    """GT boxes/labels for one COCO image, in the [x1, y1, x2, y2] absolute
    pixel format torchvision/ART expect (COCO annotations store
    [x, y, w, h]). Drops the handful of COCO annotations with zero area or
    `iscrowd=1` (crowd regions have no single box and would corrupt IoU
    matching against detector boxes).
    """
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    boxes, labels = [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["category_id"])

    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
    }


def prediction_to_coco_results(image_id: int, prediction: dict) -> list[dict]:
    """torchvision detection output (`boxes` xyxy, `labels`, `scores`, all
    tensors) -> list of pycocotools result dicts (`bbox` xywh, as required
    by `COCO.loadRes`). No score threshold here -- COCOeval needs the full
    ranked list to compute precision/recall, not just "confident" boxes.
    """
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    results = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "image_id": int(image_id),
            "category_id": int(label),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return results


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter_area / (area_a + area_b - inter_area)


def predictions_equivalent(
    pred_a: dict,
    pred_b: dict,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.5,
) -> bool:
    """True if two prediction sets describe the same detections (same
    class, IoU >= iou_thresh, both restricted to score >= score_thresh).
    Greedy one-to-one matching, symmetric in count and content.

    Implements the paper's own patch-validity filter (Section 2.1.3):
    compare predictions on the patched image against the same image with
    a black mask over the patch region. If they're equivalent, the patch
    had no effect beyond plain occlusion and gets discarded.
    """
    def boxes_labels(pred):
        scores = pred["scores"].detach().cpu().numpy()
        keep = scores >= score_thresh
        return pred["boxes"].detach().cpu().numpy()[keep], pred["labels"].detach().cpu().numpy()[keep]

    boxes_a, labels_a = boxes_labels(pred_a)
    boxes_b, labels_b = boxes_labels(pred_b)
    if len(boxes_a) != len(boxes_b):
        return False

    matched_b: set[int] = set()
    for i in range(len(boxes_a)):
        best_iou, best_j = 0.0, -1
        for j in range(len(boxes_b)):
            if j in matched_b or labels_b[j] != labels_a[i]:
                continue
            iou = _iou(boxes_a[i], boxes_b[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou < iou_thresh:
            return False
        matched_b.add(best_j)
    return True

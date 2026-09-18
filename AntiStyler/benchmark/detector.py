"""Faster R-CNN target model, wrapped for both plain torchvision use
(defense application, mAP scoring) and ART use (attack generation).

Matches AntiStyler_Demo.ipynb's own detector choice
(`fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')`)
and Table 1's "Faster RCNN" row.
"""
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


def load_detector(device: torch.device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval().to(device)
    return model


def wrap_for_art(model, device: torch.device):
    from art.estimators.object_detection import PyTorchFasterRCNN

    return PyTorchFasterRCNN(
        model=model,
        clip_values=(0, 1),
        channels_first=True,
        device_type="cpu" if device.type == "cpu" else "gpu",
    )

"""COCO val2017 loading + the train/eval split used to build the two
universal-patch attacks (Google's Adversarial Patch, DPatch).

Assumption (paper doesn't specify, see README.md "Deviations from the
paper"): AntiStyler_paper.md Section 5.1 says results are on "the COCO
dataset" and only ever mentions using each dataset's *test* set for
AntiStyler itself, since it needs no training. It does not say what data
the attacks themselves were trained on for the digital (COCO) experiments,
and no COCO train2017 download is assumed here. Google's Adversarial Patch
and DPatch are both trained once on a batch of images and then pasted onto
new images at eval time (that's the whole point of a "universal" patch);
M-PGD is inherently per-image and needs no training set at all. To avoid
downloading the 18GB train2017 set and to avoid any test-set leakage, we
split the *val2017* set itself into a disjoint ATTACK_TRAIN_SIZE-image
subset (patch training) and use the remaining images for evaluation.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
VAL_IMAGES_DIR = DATA_DIR / "val2017"
ANNOTATIONS_PATH = DATA_DIR / "annotations" / "instances_val2017.json"

SPLIT_SEED = 42
ATTACK_TRAIN_SIZE = 200


def load_coco_val2017():
    from pycocotools.coco import COCO

    return COCO(str(ANNOTATIONS_PATH))


def get_split(coco, attack_train_size: int = ATTACK_TRAIN_SIZE, eval_size: int | None = None):
    """Deterministic (seeded) split of val2017 image ids into a
    patch-training subset and an evaluation subset. `eval_size` caps the
    evaluation subset (None = use every remaining image); intended for
    pilot runs.
    """
    all_ids = sorted(coco.getImgIds())
    rng = random.Random(SPLIT_SEED)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    train_ids = shuffled[:attack_train_size]
    remaining = shuffled[attack_train_size:]
    eval_ids = remaining if eval_size is None else remaining[:eval_size]
    return train_ids, eval_ids


def image_path_for(coco, image_id: int) -> Path:
    info = coco.loadImgs([image_id])[0]
    return VAL_IMAGES_DIR / info["file_name"]

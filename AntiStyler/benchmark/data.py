"""COCO val2017 loading and the candidate-image stream used to generate
per-image adversarial patches for Table 1.

Per CVPR 2026 supplementary material (Section 2.1.3, "Adversarial Patch
Attacks"): "For the M-PGD, DPatch, and Google adversarial patch attacks
against the COCO dataset... we generated ~300 patches for each attack."
Patches are generated per-image (one patch trained per image, placed on
that image's own highest-confidence detection), not one universal patch
shared across a training/eval split -- there is no train/eval split in
the paper's methodology. Some candidate images get filtered out (patch
had no adversarial effect beyond plain occlusion, checked via a
black-mask comparison -- see coco_utils.predictions_equivalent), so we
need a candidate stream larger than 300 per attack to reach ~300 valid
results.
"""
from __future__ import annotations

import random
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
VAL_IMAGES_DIR = DATA_DIR / "val2017"
ANNOTATIONS_PATH = DATA_DIR / "annotations" / "instances_val2017.json"

CANDIDATE_SEED = 42


def load_coco_val2017():
    from pycocotools.coco import COCO

    return COCO(str(ANNOTATIONS_PATH))


def get_candidate_ids(coco, seed: int = CANDIDATE_SEED) -> list[int]:
    """Deterministic (seeded) shuffle of every val2017 image id, to draw
    from one at a time per attack until ~300 valid patches are generated.
    """
    all_ids = sorted(coco.getImgIds())
    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    return shuffled


def image_path_for(coco, image_id: int) -> Path:
    info = coco.loadImgs([image_id])[0]
    return VAL_IMAGES_DIR / info["file_name"]

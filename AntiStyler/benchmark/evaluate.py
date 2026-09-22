"""Table 1 benchmark: Undefended vs AntiStyler (Ours) on Faster R-CNN,
against Google's Adversarial Patch, M-PGD, and DPatch, on COCO val2017.

Methodology, per the CVPR 2026 paper's main text (Section 5.1) and
supplementary material (Section 2.1.3), fetched directly from
openaccess.thecvf.com/content/CVPR2026/ -- NOT the demo notebook's toy
attack cell, which an earlier version of this script wrongly sourced
hyperparameters from:

- ~300 patches generated PER ATTACK, one patch trained per candidate
  image (no universal patch, no train/eval split) -- see attacks.py.
- Each patch is placed on the candidate image's own highest-confidence
  detection, sized to alpha*min(bbox_w, bbox_h), alpha in [30%, 50%].
- Candidates where the patch has no effect beyond plain occlusion
  (predictions with the real patch vs. a black mask over the same region
  are equivalent) are discarded; keep drawing candidates until ~300 valid
  ones are collected per attack.
- Reported "benign"/"adv"/"mean" mAP@0.5 is computed over exactly that
  set of ~300 valid images (the paper's own benign/adv/mean protocol,
  Section 5.2, citing KDAT -- the black-mask filter IS how "attack
  succeeded on this image" is operationalized here, not a separate
  GT-based check).

Known gotcha (see README.md): ART's PyTorchFasterRCNN.loss_gradient()
leaves the underlying torchvision model in .train() mode. Every detection
forward pass used for actual scoring calls detector.eval() first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

import attacks
import config
from antistyler_core import AntiStyler, load_image
from coco_utils import (
    ground_truth_for_image,
    prediction_to_coco_results,
    predictions_equivalent,
)
from data import get_candidate_ids, image_path_for, load_coco_val2017
from detector import load_detector, wrap_for_art

TARGET_VALID_PATCHES = 300  # paper: "we generated ~300 patches for each attack"
CANDIDATE_LIMIT_MULTIPLIER = 5  # safety cap: stop after this many x the target, even if short


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attacks", nargs="+", default=["google", "dpatch", "mpgd"],
                    choices=["google", "dpatch", "mpgd"])
    p.add_argument("--target-valid", type=int, default=TARGET_VALID_PATCHES)
    p.add_argument("--max-iter", type=int, default=attacks.ART_MAX_ITER,
                    help="training iterations for Google Patch/DPatch (ART default: 500) "
                         "and for M-PGD (paper gives no number; matched to the ART-default budget)")
    p.add_argument("--pilot", action="store_true",
                    help="tiny end-to-end smoke run: 5 valid patches, 10 training "
                         "iterations -- to validate the pipeline, NOT to produce a "
                         "reportable number")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.pilot:
        args.target_valid = 5
        args.max_iter = 10
    return args


def detect(detector, image: torch.Tensor) -> dict:
    """One image -> torchvision prediction dict. Always re-syncs eval mode
    first (see module docstring's gotcha note)."""
    detector.eval()
    with torch.no_grad():
        return detector([image.squeeze(0)])[0]


def generate_adversarial_image(attack_name, art_detector, image, detection, patch_params, ground_truth, max_iter):
    size, shape, alpha = patch_params
    box = detection["box"]
    if attack_name == "google":
        patch, mask = attacks.generate_google_patch(art_detector, image, size, shape, max_iter=max_iter)
        return attacks.paste_patch(image, patch, mask, box, alpha)
    if attack_name == "dpatch":
        patch, mask = attacks.generate_dpatch(art_detector, image, size, max_iter=max_iter)
        return attacks.paste_patch(image, patch, mask, box, alpha)
    if attack_name == "mpgd":
        return attacks.generate_mpgd(art_detector, image, ground_truth, box, alpha, shape, max_iter=max_iter)
    raise ValueError(attack_name)


def compute_map50(coco_gt, results: list[dict], image_ids: list[int]) -> float:
    if not results:
        print("  [warn] no detections at all for this split -- reporting mAP@0.5 = 0.0")
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[1])  # AP @ IoU=0.50


def run_attack(attack_name, coco, candidate_ids, detector, defense, art_detector, args, device, rng):
    print(f"\n=== {attack_name} ===")

    results = {
        "undefended_benign": [], "undefended_adv": [],
        "antistyler_benign": [], "antistyler_adv": [],
    }
    valid_ids = []
    checked = 0
    candidate_limit = args.target_valid * CANDIDATE_LIMIT_MULTIPLIER

    for image_id in candidate_ids:
        if len(valid_ids) >= args.target_valid or checked >= candidate_limit:
            break
        checked += 1

        ground_truth = ground_truth_for_image(coco, image_id)
        if len(ground_truth["boxes"]) == 0:
            continue

        image = load_image(str(image_path_for(coco, image_id)), device)
        detection = attacks.top_confidence_detection(detector, image)
        if detection is None:
            continue

        patch_params = attacks.sample_patch_params(rng)
        adv_image = generate_adversarial_image(attack_name, art_detector, image, detection,
                                                patch_params, ground_truth, args.max_iter)
        detector.eval()  # attack generation may have left the model in train() mode

        size, shape, alpha = patch_params
        black_masked = attacks.paste_black_mask(image, detection["box"], alpha, shape)
        adv_pred_undef = detect(detector, adv_image)
        black_pred_undef = detect(detector, black_masked)
        if predictions_equivalent(adv_pred_undef, black_pred_undef):
            continue  # no adversarial effect beyond occlusion -- paper's own filter (Sec 2.1.3)

        valid_ids.append(image_id)
        benign_pred_undef = detect(detector, image)
        defended_benign_image = defense.apply(image, device)
        defended_adv_image = defense.apply(adv_image, device)
        benign_pred_def = detect(detector, defended_benign_image)
        adv_pred_def = detect(detector, defended_adv_image)

        results["undefended_benign"] += prediction_to_coco_results(image_id, benign_pred_undef)
        results["undefended_adv"] += prediction_to_coco_results(image_id, adv_pred_undef)
        results["antistyler_benign"] += prediction_to_coco_results(image_id, benign_pred_def)
        results["antistyler_adv"] += prediction_to_coco_results(image_id, adv_pred_def)

    print(f"  {len(valid_ids)}/{args.target_valid} valid patches collected "
          f"({checked} candidates checked, {checked - len(valid_ids)} filtered/skipped)")

    scores = {}
    for method in ("undefended", "antistyler"):
        benign_map = compute_map50(coco, results[f"{method}_benign"], valid_ids)
        adv_map = compute_map50(coco, results[f"{method}_adv"], valid_ids)
        scores[method] = {
            "benign": benign_map,
            "adv": adv_map,
            "mean": (benign_map + adv_map) / 2,
        }
    return scores, results, valid_ids


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (" [PILOT RUN -- numbers below are not reportable]" if args.pilot else ""))

    coco = load_coco_val2017()
    candidate_ids = get_candidate_ids(coco, seed=args.seed)
    print(f"candidate pool: {len(candidate_ids)} images, target {args.target_valid} valid patches/attack")

    detector = load_detector(device)
    art_detector = wrap_for_art(detector, device)

    backbone = config.build_backbone(device)
    defense = AntiStyler(backbone, config.CONTENT_LAYERS, config.STYLE_LAYERS,
                          config.CONTENT_WEIGHT, config.STYLE_WEIGHT, config.NUM_STEPS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for attack_name in args.attacks:
        rng = np.random.default_rng(args.seed)  # same patch-param sequence per attack, reproducible
        t0 = time.time()
        scores, raw_results, valid_ids = run_attack(
            attack_name, coco, candidate_ids, detector, defense, art_detector, args, device, rng)
        all_scores[attack_name] = scores
        elapsed = time.time() - t0
        print(f"  Undefended : benign={scores['undefended']['benign']:.3f}  "
              f"adv={scores['undefended']['adv']:.3f}  mean={scores['undefended']['mean']:.3f}")
        print(f"  AntiStyler : benign={scores['antistyler']['benign']:.3f}  "
              f"adv={scores['antistyler']['adv']:.3f}  mean={scores['antistyler']['mean']:.3f}")
        print(f"  ({elapsed:.1f}s)")

        with open(out_dir / f"{attack_name}_raw_results.json", "w") as f:
            json.dump({"valid_ids": valid_ids, **raw_results}, f)

    with open(out_dir / "scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nWrote scores to {out_dir / 'scores.json'}")


if __name__ == "__main__":
    main()

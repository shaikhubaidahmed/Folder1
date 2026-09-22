"""Table 1's three digital patch attacks, per-image, matching the actual
methodology described in the CVPR 2026 paper's supplementary material
(Section 2.1.3, "Adversarial Patch Attacks") and main paper (Section 5.1)
-- not the demo notebook's toy attack cell, which we initially (wrongly)
sourced hyperparameters from. Key facts from the primary source (both PDFs
fetched directly from openaccess.thecvf.com/content/CVPR2026/):

- "Each attack was configured according to the default settings in the
  Adversarial Robustness Toolbox (ART) library... creating both square-
  and circle-shaped patches. The original patch sizes... were set to
  {100,120,150}." -- so Google Patch/DPatch use ART's own defaults
  (learning_rate=5.0, max_iter=500), not custom-tuned values.
- "the patch was placed on the object with the highest confidence score,
  with an adjusted patch size of alpha*min(BB_width, BB_height)... alpha
  is a dynamic coefficient in the range 30%-50%" -- patches are pasted at
  a size relative to the specific target object's own bounding box, not
  at a fixed size/location.
- "we generated ~300 patches for each attack" -- one patch trained PER
  IMAGE (not a universal patch trained once and reused across an eval
  split); see data.py's candidate-stream docstring.
- "images in which the patch only caused partial occlusion without any
  adversarial effect" are excluded, checked by comparing predictions with
  and without a black mask over the patch region -- see
  coco_utils.predictions_equivalent, used by evaluate.py's generation
  loop, not here.
- M-PGD (supplementary Eq. 10): P^{t+1} = clip(P^t + alpha*grad_P L, 0, 1)
  -- plain (non-sign) gradient ascent directly on the patch region, no
  epsilon-ball constraint at all. This is NOT the small-perturbation L_inf
  PGD our first implementation used; M-PGD is architecturally a patch
  attack like the other two, just optimized via a different (untargeted,
  ground-truth-loss) objective instead of ART's targeted classification
  loss. Neither PDF gives a numeric step size or iteration count for this
  equation -- see masked_pgd's own docstring for how that gap is filled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PATCH_BASE_SIZES = (100, 120, 150)  # paper: "original patch sizes... {100,120,150}"
PATCH_SHAPES = ("square", "circle")  # paper: "creating both square- and circle-shaped patches"
ALPHA_RANGE = (0.30, 0.50)  # paper: "alpha is a dynamic coefficient in the range 30%-50%"
PERSON_LABEL = 1

ART_LEARNING_RATE = 5.0  # ART's own default for both AdversarialPatchPyTorch and DPatch
ART_MAX_ITER = 500       # ART's own default for both


def sample_patch_params(rng: np.random.Generator) -> tuple[int, str, float]:
    """Paper doesn't state the selection policy across {100,120,150} x
    {square,circle} x alpha in [30%,50%] -- sampled uniformly per image,
    a documented assumption, not a paper-sourced rule."""
    size = int(rng.choice(PATCH_BASE_SIZES))
    shape = str(rng.choice(PATCH_SHAPES))
    alpha = float(rng.uniform(*ALPHA_RANGE))
    return size, shape, alpha


def top_confidence_detection(detector, image: torch.Tensor) -> dict | None:
    """The object the patch gets placed on: "the object with the highest
    confidence score" (paper, Section 2.1.3). Returns None if the
    undefended detector finds nothing in this image at all.
    """
    detector.eval()
    with torch.no_grad():
        pred = detector([image.squeeze(0)])[0]
    if pred["scores"].numel() == 0:
        return None
    idx = int(pred["scores"].argmax())
    return {
        "box": pred["boxes"][idx].detach().cpu().numpy(),
        "label": int(pred["labels"][idx]),
    }


def circular_mask(size: int, sharpness: int = 40) -> np.ndarray:
    """Exact same formula as ART's own AdversarialPatchPyTorch._get_circular_patch_mask
    (read from ART's source), so a hand-rolled M-PGD circular patch uses
    the identical circle shape ART itself uses for Google Patch."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x_grid, y_grid = np.meshgrid(x, y, sparse=True)
    z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness
    mask = 1 - np.clip(z_grid, -1, 1)
    return np.broadcast_to(mask[None, :, :], (3, size, size)).astype(np.float32)


def _paste_geometry(image_shape, box: np.ndarray, alpha: float) -> tuple[int, int, int, int, int]:
    """Target paste size (alpha * min(bbox_w, bbox_h)) and its centered
    placement, shifted (not cropped) to stay fully on-canvas. `target` is
    capped to the smaller image dimension, so a valid (x0, y0, x0+target,
    y0+target) box always exists -- callers can assume the returned
    region is always exactly target x target, never truncated.
    Returns (x0, y0, x1, y1, target_size) in the full image's pixel
    coordinates."""
    _, _, h, w = image_shape
    x1b, y1b, x2b, y2b = box
    bw, bh = max(x2b - x1b, 1.0), max(y2b - y1b, 1.0)
    target = max(4, min(int(round(alpha * min(bw, bh))), w, h))
    cx, cy = (x1b + x2b) / 2.0, (y1b + y2b) / 2.0
    x0 = int(round(cx - target / 2))
    y0 = int(round(cy - target / 2))
    x0 = min(max(0, x0), w - target)
    y0 = min(max(0, y0), h - target)
    return x0, y0, x0 + target, y0 + target, target


def paste_patch(image: torch.Tensor, patch: np.ndarray, mask: np.ndarray, box: np.ndarray, alpha: float) -> torch.Tensor:
    """Resize a trained (size, size) patch + its shape mask to
    alpha*min(bbox_w, bbox_h) and blend it onto `image`, centered on
    `box`, clipped to the canvas. Used for Google Patch and DPatch (whose
    ART classes train the patch with their own internal random
    placement/augmentation, per the paper using "default settings"); the
    final evaluation-time placement is done here, manually, to match the
    paper's specific "place on the top-confidence object, sized to its
    bbox" rule -- ART's own apply_patch only supports random-within-mask
    placement, not an exact deterministic location.
    """
    x0, y0, x1, y1, target = _paste_geometry(image.shape, box, alpha)

    patch_t = torch.from_numpy(patch).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask).unsqueeze(0).float()
    patch_r = F.interpolate(patch_t, size=(target, target), mode="bilinear", align_corners=False)[0]
    mask_r = F.interpolate(mask_t, size=(target, target), mode="bilinear", align_corners=False)[0]

    out = image.clone()
    p = patch_r.to(image.device, image.dtype)
    m = mask_r.to(image.device, image.dtype)
    out[:, :, y0:y1, x0:x1] = out[:, :, y0:y1, x0:x1] * (1 - m) + p * m
    return out


def paste_black_mask(image: torch.Tensor, box: np.ndarray, alpha: float, shape: str) -> torch.Tensor:
    """Same placement geometry as paste_patch, but with a solid black
    region instead of the trained patch -- the paper's own patch-validity
    check (Section 2.1.3): "comparing the predictions on the attacked
    image with and without a black mask on the patched area."
    """
    x0, y0, x1, y1, target = _paste_geometry(image.shape, box, alpha)
    mask = circular_mask(target) if shape == "circle" else np.ones((3, target, target), dtype=np.float32)
    m = torch.from_numpy(mask).to(image.device, image.dtype)
    out = image.clone()
    out[:, :, y0:y1, x0:x1] = out[:, :, y0:y1, x0:x1] * (1 - m)
    return out


def generate_google_patch(art_detector, image: torch.Tensor, size: int, shape: str,
                           max_iter: int = ART_MAX_ITER, learning_rate: float = ART_LEARNING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Brown et al.'s Adversarial Patch (ART's AdversarialPatchPyTorch),
    trained on this single image with ART's own default hyperparameters
    and its own internal random location/rotation/scale augmentation
    (left at ART's defaults, per the paper's "default settings" -- not
    disabled the way our first, incorrect implementation did). Returns
    the raw (patch, shape_mask) at `size`x`size`; final placement at the
    target object's location/scale is done by paste_patch, separately.
    """
    from art.attacks.evasion import AdversarialPatchPyTorch

    attack = AdversarialPatchPyTorch(
        estimator=art_detector,
        patch_shape=(3, size, size),
        patch_type=shape,
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=1,
        targeted=True,
        verbose=False,
    )
    x = image.detach().cpu().numpy()
    target = [{
        "boxes": np.array([[0.0, 0.0, float(size), float(size)]], dtype=np.float32),
        "labels": np.array([PERSON_LABEL], dtype=np.int64),
        "scores": np.array([1.0], dtype=np.float32),
    }]
    patch, patch_mask = attack.generate(x=x, y=target)
    return patch, patch_mask


def generate_dpatch(art_detector, image: torch.Tensor, size: int,
                     max_iter: int = ART_MAX_ITER, learning_rate: float = ART_LEARNING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Liu et al.'s DPatch (ART's DPatch), ART default hyperparameters.
    DPatch has no patch_type/shape option in ART (always rectangular) --
    unlike Google Patch, there's no circle variant to choose here; that's
    a real constraint of ART's DPatch implementation, not an omission.
    """
    from art.attacks.evasion import DPatch

    attack = DPatch(
        estimator=art_detector,
        patch_shape=(3, size, size),
        learning_rate=learning_rate,
        max_iter=max_iter,
        batch_size=1,
        verbose=False,
    )
    x = image.detach().cpu().numpy()
    patch = attack.generate(x=x, target_label=PERSON_LABEL)
    mask = np.ones((3, size, size), dtype=np.float32)
    return patch, mask


# M-PGD's step size: supplementary Eq. 10 (P += alpha*grad_P L, clipped to
# [0,1], no epsilon-ball) gives no numeric alpha or iteration count in
# either the main paper or supplement. Calibrated empirically (not
# paper-sourced): measured Faster R-CNN's loss_gradient magnitude on a
# real COCO image (mean|grad| ~= 3e-4, max|grad| ~= 0.04) and picked
# alpha so max_iter steps of accumulated gradient ascent meaningfully
# traverse the [0,1] pixel range without saturating in only a few steps.
MPGD_STEP_SIZE = 3.0
MPGD_MAX_ITER = ART_MAX_ITER  # matched to Google Patch/DPatch's own ART-default budget


def generate_mpgd(art_detector, image: torch.Tensor, ground_truth: dict, box: np.ndarray,
                   alpha_size: float, shape: str, step_size: float = MPGD_STEP_SIZE,
                   max_iter: int = MPGD_MAX_ITER, seed: int | None = None) -> torch.Tensor:
    """M-PGD, supplementary Eq. 10: P^{t+1} = clip(P^t + alpha*grad_P L, 0, 1),
    untargeted (maximizes the detector's loss w.r.t. the image's real
    ground-truth boxes/labels -- the paper states M-PGD "can be targeted
    or untargeted"; kept untargeted here, consistent with standard PGD's
    own objective and this implementation's earlier choice).

    Unlike Google Patch/DPatch (whose ART classes train with their own
    internal random location sampling, then get manually placed at the
    target location afterward), Eq. 10 has no expectation over location
    (unlike Google Patch's Eq. 9, which explicitly is) -- so M-PGD is
    optimized directly, in place, at its final target location and size
    from the first step.
    """
    device = image.device
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    x0, y0, x1, y1, target = _paste_geometry(image.shape, box, alpha_size)

    if shape == "circle":
        mask_region = torch.from_numpy(circular_mask(target)).to(device)
    else:
        mask_region = torch.ones((3, target, target), device=device)

    patch = torch.rand((1, 3, target, target), generator=rng).to(device)

    def assemble(p):
        out = image.clone()
        out[:, :, y0:y1, x0:x1] = image[:, :, y0:y1, x0:x1] * (1 - mask_region) + p * mask_region
        return out

    x_adv = assemble(patch)
    target_dict = [{"boxes": ground_truth["boxes"], "labels": ground_truth["labels"]}]

    for _ in range(max_iter):
        grad = art_detector.loss_gradient(x=x_adv.detach(), y=target_dict)
        region_grad = grad[:, :, y0:y1, x0:x1]
        patch = (patch + step_size * region_grad).clamp(0, 1)
        x_adv = assemble(patch).detach()

    return x_adv

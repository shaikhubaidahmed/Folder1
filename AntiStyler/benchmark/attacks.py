"""Table 1's three digital patch attacks: Google's Adversarial Patch
(Brown et al. [3]), DPatch (Liu et al. [29]), and M-PGD (AntiStyler's own
masked-PGD adaptation of Madry et al.'s PGD [30] to a patch-shaped region
-- "M-PGD" is the paper's name for it, not a separate cited method; ref
[30] is the original untargeted PGD paper, and "masked" describes
AntiStyler's own restriction of the perturbation to a fixed region).

Hyperparameters NOT stated in the paper's main text (patch size, learning
rate, iteration count, epsilon) are deferred to "the supplementary
material" (Section 5.1), which is not available to us. Every default
below is sourced from either (a) the attack's own original paper, or (b)
the AntiStyler demo notebook's own attack cell (100x100 patch, target
class "Person"), and is clearly NOT claimed to be the paper's exact
setting. This is a documented assumption, not a fabricated number --
see benchmark/README.md.

Google's Adversarial Patch and DPatch are trained once on a batch of
images (`attack_train_ids` in data.py) and then pasted onto new images at
eval time with no further optimization -- that's what makes them
"universal" patches (Brown et al. Section 3; Liu et al. Section 3). M-PGD
is inherently per-image (Madry et al.'s threat model perturbs each input
individually) and is generated directly on each evaluation image.
"""
from __future__ import annotations

import numpy as np
import torch


PATCH_SIZE = 100  # matches the demo notebook's own attack_config['patch_size']


def _to_numpy_batch(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()


def train_google_patch(art_detector, train_images: np.ndarray, max_iter: int = 200):
    """Brown et al.'s Adversarial Patch, ART's AdversarialPatchPyTorch.
    Targeted (the attack's own defining property: force the target class
    to be predicted with high confidence), square patch, no rotation/scale
    augmentation (kept off since ART's default random rotation/scaling is
    meant for the classification setting's patch-in-photo scenario, not
    this paper's fixed digital patch placement).
    """
    from art.attacks.evasion import AdversarialPatchPyTorch

    attack = AdversarialPatchPyTorch(
        estimator=art_detector,
        rotation_max=0.0,
        scale_min=1.0,
        scale_max=1.0,
        patch_shape=(3, PATCH_SIZE, PATCH_SIZE),
        patch_type="square",
        learning_rate=0.03,
        max_iter=max_iter,
        batch_size=8,
        targeted=True,
        verbose=True,
    )
    # target_label=1 ("person"), matching the demo notebook's own choice
    # of target_label="Person" for its analogous toy attack.
    n = train_images.shape[0]
    target = [{
        "boxes": np.array([[0.0, 0.0, float(PATCH_SIZE), float(PATCH_SIZE)]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "scores": np.array([1.0], dtype=np.float32),
    } for _ in range(n)]
    patch, patch_mask = attack.generate(x=train_images, y=target)
    return attack, patch, patch_mask


def apply_google_patch(attack, image: torch.Tensor) -> torch.Tensor:
    x = _to_numpy_batch(image)
    x_patched = attack.apply_patch(x, scale=1.0)
    return torch.from_numpy(x_patched).to(image.device, image.dtype)


def train_dpatch(art_detector, train_images: np.ndarray, max_iter: int = 200):
    """Liu et al.'s DPatch: a targeted "creation" attack -- the patch is
    optimized so the detector predicts `target_label` AT THE PATCH'S OWN
    location, not at some other object's box (Liu et al. Section 3;
    confirmed against ART's own DPatch source, which places `target_label`
    at the patch's sampled bounding box during training).
    """
    from art.attacks.evasion import DPatch

    attack = DPatch(
        estimator=art_detector,
        patch_shape=(PATCH_SIZE, PATCH_SIZE, 3),
        learning_rate=1.0,
        max_iter=max_iter,
        batch_size=8,
        verbose=True,
    )
    attack.generate(x=train_images, target_label=1)  # 1 = "person"
    return attack


def apply_dpatch(attack, image: torch.Tensor) -> torch.Tensor:
    x = _to_numpy_batch(image)
    x_patched = attack.apply_patch(x)
    return torch.from_numpy(x_patched).to(image.device, image.dtype)


def masked_pgd(art_detector, image: torch.Tensor, ground_truth: dict, eps: float = 16 / 255,
               eps_step: float = 2 / 255, max_iter: int = 40, patch_size: int = PATCH_SIZE,
               seed: int | None = None) -> torch.Tensor:
    """M-PGD: Madry et al.'s PGD (untargeted -- maximize the detector's
    loss w.r.t. the image's true ground-truth boxes/labels, the exact
    Madry et al. objective, not a targeted creation attack), with the
    perturbation masked to a single fixed patch-shaped region rather than
    the whole image (the "masked" in "M-PGD").

    ART's ProjectedGradientDescentPyTorch only accepts classifier
    estimators (verified via its constructor signature), not object
    detectors, so this is a direct, from-scratch implementation of
    Madry et al.'s Algorithm (random start in the eps-ball, then
    `max_iter` steps of x <- clip_eps(x + eps_step * sign(grad_x L(x, y))))
    against `art_detector.loss_gradient`, which IS supported for object
    detectors -- this keeps the attack's threat model (L_inf, sign-gradient
    ascent, eps-ball projection) faithful to the original paper while
    using ART only for the gradient computation, not a repurposed
    classifier attack.
    """
    device = image.device
    _, _, h, w = image.shape
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    y0 = max(0, (h - patch_size) // 2)
    x0 = max(0, (w - patch_size) // 2)
    mask = torch.zeros_like(image)
    mask[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = 1.0

    x_orig = image.clone()
    delta = (torch.rand(image.shape, generator=rng).to(device) * 2 - 1) * eps
    x_adv = (x_orig + delta * mask).clamp(0, 1)

    target = [{
        "boxes": ground_truth["boxes"],
        "labels": ground_truth["labels"],
    }]

    for _ in range(max_iter):
        grad = art_detector.loss_gradient(x=x_adv.detach().cpu().numpy(), y=target)
        grad = torch.from_numpy(grad).to(device)
        x_adv = x_adv + eps_step * grad.sign() * mask
        perturbation = (x_adv - x_orig).clamp(-eps, eps)
        x_adv = (x_orig + perturbation * mask).clamp(0, 1).detach()

    return x_adv

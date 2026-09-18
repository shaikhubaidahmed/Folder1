# Table 1 benchmark harness (work in progress)

Goal: reproduce AntiStyler_paper.md's Table 1 -- specifically the
**Undefended** and **AntiStyler (Ours)** rows on **Faster R-CNN**, across
the three digital attacks (Google's Adversarial Patch, M-PGD, DPatch) on
COCO. The other five defense rows (ObjectSeeker, PAD, DIFFender, NutNet,
KDAT) are out of scope here -- reproducing them would require sourcing
and running each paper's own released code, not reimplementing them from
this paper's description.

**Status: pipeline built and each component individually smoke-tested
(correct shapes, no crashes, real forward/backward passes verified), but
no end-to-end pilot or full run has completed yet.** No mAP numbers exist
yet -- nothing in this benchmark should be treated as a result until an
actual run finishes and is reported.

## What's implemented

- `data.py` -- COCO val2017 loading (via pycocotools) and a seeded
  train/eval split of val2017 itself (200 images for attack training, the
  rest for evaluation). Verified: COCO's category ids (1-90, with gaps)
  match torchvision's Faster R-CNN label ids directly, confirmed by
  running the real detector on a real COCO image and checking every
  predicted label is a valid COCO category id.
- `detector.py` -- `fasterrcnn_resnet50_fpn` (COCO-pretrained), matching
  AntiStyler_Demo.ipynb's own detector choice, wrapped for ART via
  `PyTorchFasterRCNN`.
- `antistyler_core.py` -- the paper's own AntiStyler/AntiStyle/ContentLoss/
  StyleLoss classes, extracted verbatim from `AntiStyler_Demo.ipynb` (not
  hand-copied) so the reference implementation can't silently drift from
  the notebook. **Verified exact parity**: ran both the notebook's inline
  code and this extracted module on the same image/seed --
  `torch.equal(...) == True` on the defended output.
  The only change is gating the notebook's 5 `imshow()` calls per
  `.apply()` behind `visualize=False` (default), since calling matplotlib
  5000+ times is unusable for a batch run; the tensor computation itself
  is untouched.
- `attacks.py` -- the three Table 1 attacks:
  - **Google's Adversarial Patch** (Brown et al. [3]): ART's
    `AdversarialPatchPyTorch`, targeted, trained once on the 200-image
    training split, then pasted onto each eval image.
  - **DPatch** (Liu et al. [29]): ART's `DPatch`, also trained once on the
    same 200-image split (a targeted "creation" attack -- the patch is
    optimized to make the detector predict the target class *at the
    patch's own location*, confirmed against ART's `DPatch` source).
  - **M-PGD**: hand-implemented, *not* via ART's PGD class. ART's
    `ProjectedGradientDescentPyTorch` only accepts classifier estimators
    (checked its constructor signature directly), not object detectors,
    so it can't be used here. Implemented from scratch instead: Madry et
    al.'s actual PGD algorithm (random start in the L_inf eps-ball, then
    sign-gradient ascent + eps-ball projection, `max_iter` steps) against
    `art_detector.loss_gradient` (which *does* support object detectors),
    masked to a fixed patch-shaped region. This is untargeted (maximizes
    loss on the image's true ground-truth boxes/labels -- Madry et al.'s
    own objective), unlike Google's patch and DPatch which are targeted.

## Deviations / assumptions (not claimed to match the paper exactly)

The paper's Section 5.1 defers attack hyperparameters (patch size,
learning rate, iteration counts, epsilon) to **supplementary material we
don't have**. Every default in `attacks.py` is sourced from either the
attack's own original paper/ART's defaults, or the demo notebook's own
toy attack cell (`patch_size=100`, `target_label="Person"`) -- never
invented to match a Table 1 number. Specifically:

- Attack training data: the paper never states what data trains the
  digital attacks (only that AntiStyler itself uses "each dataset's test
  set" since it needs no training). To avoid downloading COCO's 18GB
  train2017 set and to avoid test-set leakage, we split *val2017 itself*
  into a 200-image training subset and a disjoint evaluation subset
  (seeded, `data.py:SPLIT_SEED=42`).
- Patch size 100x100, target class "person" (label id 1): taken from the
  demo notebook's own attack cell, not from Table 1's methodology.
- Google Patch / DPatch `max_iter=200`, `batch_size=8`; M-PGD
  `eps=16/255`, `eps_step=2/255`, `max_iter=40`: reasonable defaults for
  each attack's own literature, not paper-sourced numbers.
- The paper's "benign/adv/mean" evaluation protocol (Section 5.2, citing
  KDAT [21]) is **not yet implemented**: it requires an operational
  definition of "attack succeeded on this image" that the paper doesn't
  spell out. Working definition to be implemented: an attack succeeds on
  an image if at least one ground-truth object the undefended detector
  correctly detected (IoU >= 0.5, correct class) on the benign image is no
  longer correctly detected on the attacked image. This needs to be
  reviewed before it's used to produce any reported number.

## Measured compute cost (Apple M5, CPU only -- no CUDA on this machine)

Real measurements, not estimates, taken 2026-09-18 on a single eval image
(`AntiStyler/benchmark/data/val2017/000000460147.jpg`, 424x640) and an
8-image training mini-batch:

| Step | Measured cost |
|---|---|
| Google Patch training, 1 iteration (batch=8) | ~13.7s |
| M-PGD, 1 gradient step (`loss_gradient` call) | ~1.47s |
| AntiStyler.apply, 1 image | ~2.19s |
| Faster R-CNN forward pass, 1 image (eval mode) | ~0.48s |

Extrapolated to the originally-planned full scale (200-image attack
training x200 iterations, 4800-image eval set, 3 attacks): **~6-7 days of
continuous single-threaded CPU compute** -- Google Patch training alone
is ~19 hours, M-PGD generation across all eval images is ~78 hours. This
is why the full run was paused rather than launched: decided with the
user to continue this benchmark on a remote GPU machine instead of
running it for a week on a laptop CPU.

**Gotcha found and worth keeping in mind on the GPU machine too**: ART's
`PyTorchFasterRCNN.loss_gradient()` (used internally by all three attacks)
puts the underlying torchvision model into `.train()` mode to compute a
loss dict, and does not restore it to `.eval()` afterward. Any detection
forward pass used for actual evaluation (benign detection, defended
detection, mAP scoring) must call `model.eval()` again first, or it will
crash with `"targets should not be none when in training mode"` -- this
bit the timing measurements above the first time and needs to be handled
explicitly in `evaluate.py` (not yet written), e.g. by re-calling
`.eval()` right before every eval-purpose forward pass.

## Not yet built

- `evaluate.py`: the actual mAP@0.5 scoring loop (pycocotools-based) and
  the benign/adv/mean protocol.
- A pilot run end-to-end (train patches on a few images, attack a few
  eval images, defend, score) to validate correctness before any
  full-scale run.
- DPatch's per-iteration training cost was not separately measured (only
  Google Patch's was); assume the same order of magnitude, not verified.

## Setup

```bash
python3 -m pip install --user adversarial-robustness-toolbox pycocotools
```

COCO val2017 images + `instances_val2017.json` must be present under
`AntiStyler/benchmark/data/val2017/` and
`AntiStyler/benchmark/data/annotations/` (gitignored -- ~1.6GB, download
separately: `http://images.cocodataset.org/zips/val2017.zip` and
`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`).

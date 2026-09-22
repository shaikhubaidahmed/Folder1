# Table 1 benchmark harness

Goal: reproduce AntiStyler_paper.md's Table 1 -- specifically the
**Undefended** and **AntiStyler (Ours)** rows on **Faster R-CNN**, across
the three digital attacks (Google's Adversarial Patch, M-PGD, DPatch) on
COCO. The other five defense rows (ObjectSeeker, PAD, DIFFender, NutNet,
KDAT) are out of scope here -- reproducing them would require sourcing
and running each paper's own released code, not reimplementing them from
this paper's description.

**Status: complete.** A full run (300 independently-generated adversarial
patches per attack, matching the paper's own scale) finished on GPU. See
"Results" below. The methodology was corrected midway through this
project after fetching and reading the actual CVPR 2026 paper +
supplementary material directly (openaccess.thecvf.com) -- an earlier
version of this harness had guessed hyperparameters from
`AntiStyler_Demo.ipynb`'s unrelated toy attack cell instead, which turned
out to diverge from the published methodology in several ways (see
"History" below).

## Results

mAP@0.5 (%), Faster R-CNN, our reproduction vs. the paper's Table 1:

| Attack | Source | Undef Benign | Undef Adv | AntiStyler Benign | AntiStyler Adv | Δ Adv |
|---|---|---|---|---|---|---|
| Google Patch | Paper | 51.6 | 16.6 | 51.6 | 32.5 | +15.9 |
| Google Patch | Ours | 60.7 | 55.3 | 60.6 | 53.9 | **−1.4** |
| DPatch | Paper | 49.4 | 23.0 | 48.9 | 38.3 | +15.3 |
| DPatch | Ours | 60.6 | 54.9 | 60.4 | 53.7 | **−1.2** |
| M-PGD | Paper | 45.8 | 22.3 | 45.8 | 38.0 | +15.7 |
| M-PGD | Ours | 65.8 | 32.1 | 65.7 | 51.2 | **+19.1** |

Raw per-attack COCO-format detections and scores: `results_google/`,
`results_dpatch/`, `results_mpgd/` (gitignored, regenerate by rerunning
`run_full_benchmark.sh`).

**M-PGD reproduces the paper's claim** (even slightly exceeds it).
**Google Patch and DPatch do not** -- AntiStyler shows a small
*regression* instead of the paper's ~+15pt improvement. This divergence
was investigated at length (see "Google Patch / DPatch divergence"
below) and is an honest, unresolved discrepancy, not a known bug.

## Methodology

Per the CVPR 2026 paper (Section 5.1) and its supplementary material
(Section 2.1.3), fetched directly as PDFs from
`openaccess.thecvf.com/content/CVPR2026/`:

- **~300 patches generated per attack** (`evaluate.py:TARGET_VALID_PATCHES`),
  one patch trained *per candidate image* -- not a universal patch shared
  across a train/eval split.
- **Placement**: each patch goes on its own image's highest-confidence
  detection (`attacks.top_confidence_detection`), sized to
  `alpha * min(bbox_w, bbox_h)`, `alpha` in [30%, 50%]. Base training
  size drawn from {100, 120, 150}px, shape from {square, circle} --
  matches the paper's "original patch sizes... {100,120,150}" and
  "creating both square- and circle-shaped patches."
- **Hyperparameters**: ART's own defaults (`learning_rate=5.0,
  max_iter=500`) for both `AdversarialPatchPyTorch` (Google Patch) and
  `DPatch` -- per the paper's "each attack was configured according to
  the default settings in ART" -- not the demo notebook's toy-cell values
  used in an earlier version of this harness.
- **Validity filter**: a candidate is discarded if predictions on the
  patched image are equivalent (`coco_utils.predictions_equivalent`,
  IoU>=0.5 matching) to predictions on the same image with a **black
  mask** over the patch region -- the paper's own check for "images in
  which the patch only caused partial occlusion without any adversarial
  effect." Candidates keep being drawn from a seeded shuffle of all 5000
  val2017 images (`data.get_candidate_ids`) until 300 valid ones are
  collected.
- **M-PGD** (supplementary Eq. 10): `P <- clip(P + alpha*grad_P L, 0, 1)`
  -- plain, unconstrained gradient ascent directly on the patch region,
  clipped only to the valid pixel range. No `sign()`, no epsilon-ball.
  This is architecturally a patch attack like the other two (not the
  small-perturbation L_inf PGD an earlier version of this harness
  implemented), just optimized via untargeted ground-truth loss instead
  of ART's targeted classification loss. Hand-implemented in
  `attacks.generate_mpgd` since ART's `ProjectedGradientDescentPyTorch`
  only accepts classifier estimators, not object detectors.
- AntiStyler's own hyperparameters (`config.py`) were already exactly
  right from the start: VGG19 backbone, `content_layers=['conv_4']`,
  `style_layers=['conv_1'..'conv_5']`, weight ratio 1:1000, 1 optimization
  step, padding 10 -- confirmed against the main paper's own
  "Implementation Details" paragraph (Section 4, not just the notebook).

## Remaining documented assumptions

Neither the main paper nor the supplement pins these down; the values
below are reasonable choices, not paper-sourced numbers:

- **Patch size/shape/alpha sampling policy**: uniform random per image
  across {100,120,150} x {square,circle} x alpha~U(30%,50%)
  (`attacks.sample_patch_params`). Tested varying the *target class*
  instead (see below) -- had zero effect, so this specific policy wasn't
  the source of the Google/DPatch divergence, but the sampling rule
  itself is still an assumption.
- **M-PGD step size and iteration count**: neither PDF gives a number.
  Calibrated empirically (not paper-sourced): measured Faster R-CNN's
  `loss_gradient` magnitude on a real COCO image (mean|grad|~=3e-4,
  max|grad|~=0.04) and picked `alpha=3.0`, `max_iter=500` (matched to the
  other two attacks' ART-default budget) so the accumulated gradient
  ascent meaningfully traverses [0,1] without saturating in a few steps.
- **"Mean" definition**: arithmetic mean of benign/adv mAP@0.5, not mAP
  over the pooled image set -- the paper doesn't say which.
- **Attack training data / target class**: `PERSON_LABEL=1` used as
  Google Patch/DPatch's target class for every image, taken from the demo
  notebook's own toy cell -- but empirically confirmed *not* to matter
  (see below), so this specific assumption is not load-bearing.

## Google Patch / DPatch divergence: investigation

AntiStyler shows a small **regression** (not just "no effect") against
our Google Patch/DPatch, opposite the paper's own ~+15pt improvement.
Six specific, evidence-based hypotheses were tested and each was
**refuted** -- kept here so the same dead ends aren't re-investigated:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | ART's rotation/scale training augmentation was wrongly disabled | Re-trained with ART's real defaults (`rotation_max=22.5, scale_min=0.1`) | Patch still evades AntiStyler (`torch.equal(defended, adv)==True`) |
| 2 | AntiStyler's own SR process (`num_steps=1`) is too weak to react to a real patch | Re-ran defense with `num_steps` in {1,5,20} on the same adversarial image | No change at any step count |
| 3 | Faster R-CNN checkpoint differs from the paper's | Checked `FasterRCNN_ResNet50_FPN_Weights` | Only one checkpoint (`COCO_V1`) has ever existed for this architecture -- not a variable |
| 4 | Bilinear resize (patch training size -> eval-time paste size) smooths away the patch's high-frequency structure | Measured patch saturation after bilinear vs. nearest-neighbor resize (99% -> 32-36% for bilinear, stays ~99% for nearest); re-ran end-to-end with each | Both resize modes trigger AntiStyler on the *identical* set of test images -- no effect on outcome |
| 5 | Small patches get over-masked relative to AntiStyler's fixed 11/11/51/11 enhancement kernels | Compared mean relative bbox size across helped (43.2%) / hurt (45.7%) / tied (43.7%) outcome groups on the full 300-image Google Patch run | No meaningful difference |
| 6 | Hard-coded "person" target class biases the patch's visual character | Re-generated the same 12 images' patches once targeting "person", once targeting a random COCO class | Identical outcome for every image pair -- target class makes no difference |

What *is* established (not a hypothesis, directly measured from the full
run's raw results): AntiStyler's mask fires about as often on Google
Patch (233/300 images changed something) as on M-PGD (212/300) -- this
is not a sensitivity problem. The difference is *accuracy* when it does
fire: among changed images, M-PGD's mask helps detection 94/212 times
vs. hurts 28/212 (77% helpful); Google Patch's mask helps only 21/233
times vs. hurts 39/233 (35% helpful). Something about *where* the mask
lands differs systematically between attack types, and none of the six
tests above found the mechanism. Also directly measured: real trained
Google Patch/DPatch patches are ~99-100% saturated at the extremes (0 or
1 per pixel) with *higher* adjacent-pixel variation than pure random
noise -- so "the patch is too smooth" (the original, pre-correction
theory, formed under the *old*, wrong-hyperparameter methodology) is
also not the explanation under the corrected settings.

**Honest conclusion**: this reproduction faithfully implements the
published methodology (verified against the primary-source PDFs, not
guessed) and reproduces the paper's claim for M-PGD, but not for Google
Patch/DPatch, for a reason not identified despite a real investigation.
Numbers were not tuned to match the paper -- see the six refuted
hypotheses above; nothing was adjusted post-hoc to force agreement.

## History: bugs and methodology corrections found by actually running this

- **DPatch's `patch_shape` channel order.** `DPatch.generate()` picks the
  channel index off the *estimator's* `channels_first` flag, not off
  `patch_shape`'s own layout. Our channels_first=True estimator (see
  `detector.py`) needs `patch_shape=(3, size, size)`, not the
  channels-last order ART's own docstring shows by default -- passing the
  docstring's order raised `ValueError: The color channel index of the
  images and the patch have to be identical.` Fixed in `attacks.py`.
- **The original methodology was wrong wholesale.** An earlier version of
  this harness trained one "universal" patch on a 200-image split of
  val2017 and pasted it (via ART's own `apply_patch`, mostly default
  placement) onto ~4800 held-out eval images, using hyperparameters
  (`lr=0.03`, `max_iter=200`, fixed 100x100 size) taken from
  `AntiStyler_Demo.ipynb`'s toy attack cell. Under that methodology,
  AntiStyler had *zero* measurable effect on Google Patch (4608/4619
  adversarial images were byte-identical to undefended). Reading the
  actual paper + supplement revealed the real methodology described
  above (per-image patches, ART defaults, ~300 images, black-mask
  filter, a structurally different M-PGD) -- a full rewrite of
  `data.py`, `attacks.py`, `coco_utils.py`, and `evaluate.py` followed.
  The corrected methodology raised AntiStyler's engagement rate on Google
  Patch from 0.2% to 78% of images, confirming the rewrite fixed a real
  gap -- but did not fix the net-negative outcome documented above.
- ART's `PyTorchFasterRCNN.loss_gradient()` puts the underlying
  torchvision model into `.train()` mode and does not restore
  `.eval()` afterward. Every detection forward pass used for actual
  scoring calls `detector.eval()` first (see `evaluate.py:detect`).

## What's implemented

- `data.py` -- COCO val2017 loading and a seeded candidate-image stream
  (`get_candidate_ids`) drawn from one at a time per attack until ~300
  valid patches are collected.
- `detector.py` -- `fasterrcnn_resnet50_fpn` (COCO-pretrained, the only
  checkpoint this architecture has), wrapped for ART via
  `PyTorchFasterRCNN`.
- `antistyler_core.py` -- AntiStyler/AntiStyle/ContentLoss/StyleLoss,
  extracted verbatim from `AntiStyler_Demo.ipynb`. Verified exact parity
  (`torch.equal(...)==True`) against the notebook's own inline code on
  the same image/seed. Only change: the notebook's 5 `imshow()` calls per
  `.apply()` are gated behind `visualize=False` (default).
- `attacks.py` -- per-image Google Patch / DPatch / M-PGD generation and
  placement, per "Methodology" above.
- `config.py` -- AntiStyler's hyperparameters (see "Methodology").
- `coco_utils.py` -- ground-truth extraction, COCO-format result
  conversion for `pycocotools`, and `predictions_equivalent` (the
  black-mask validity filter).
- `evaluate.py` -- orchestrates all of the above: draws candidates,
  generates/places each attack's patch, applies the black-mask filter,
  scores mAP@0.5 with `pycocotools.COCOeval` for Undefended and
  AntiStyler on both benign and adversarial images, writes
  `<out-dir>/scores.json` plus raw COCO-format detections. Has a
  `--pilot` flag (5 valid patches, 10 training iterations) for a fast
  correctness check.
- `run_full_benchmark.sh` -- runs Google Patch + DPatch concurrently,
  then M-PGD. **Note**: this GPU has no MPS, so two CUDA processes
  time-slice the same compute rather than truly parallelizing --
  measured wall-clock for Google+DPatch together was ~33-34h, not the
  ~15-20h naively expected from halving sequential time. Real per-image
  cost at `max_iter=500`, native resolution: ~112-118s (measured on a
  Quadro RTX 4000), consistent across all three attacks.

## Setup

```bash
python3 -m pip install --user -r requirements.txt
```

Pinned to what was actually run (see `requirements.txt`): `torch==2.2.1`,
`torchvision==0.17.1`, `adversarial-robustness-toolbox==1.15.1` (matches
the paper's own Section 4 ART version), `pycocotools==2.0.11`. On a
different CUDA version, install the matching `torch`/`torchvision` build
from pytorch.org first, then install the rest of `requirements.txt`.

COCO val2017 images + all six `annotations_trainval2017.zip` files must
be present under `AntiStyler/benchmark/data/val2017/` and
`AntiStyler/benchmark/data/annotations/` (gitignored -- ~1GB images +
~815MB annotations; only `instances_val2017.json` is actually used).

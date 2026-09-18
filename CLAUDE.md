# CLAUDE.md

## Project Context

This is a **research-grade Computer Vision project**, not a casual coding project. The work targets an **A*/top-tier conference** (e.g., CVPR, ICCV, ECCV, NeurIPS, ICML) and centers on **Object Detection and Adversarial Machine Learning**, including:

- Object detection architectures and training
- Adversarial attack generation (FGSM, PGD, CW, AutoAttack, adversarial patches, etc.)
- Adversarial training and robustness
- Adversarial defense mechanisms
- Robust object detection and evaluation
- Experimental methodology, ablation studies, and benchmarking
- Research code, datasets, metrics, visualization, and reproducibility

The current codebase includes the **AntiStyler** project (`AntiStyler/`) — a training-free, style-removal-based defense against adversarial patch attacks on object detectors, with an accompanying paper transcription (`AntiStyler/AntiStyler_paper.md`) and figures (`AntiStyler/paper_figures/`).

## Research Standard

Optimize all work in this priority order:

**Scientific correctness → Reproducibility → Rigorous evaluation → Novelty validation → Clean implementation → Performance**

Never optimize for producing code quickly at the expense of research validity.

## Operating Rules

1. Treat this as a **research-grade project**. Correctness, reproducibility, and experimental validity take priority over speed.
2. **Verify everything before writing code.** Do not blindly implement assumptions — check the underlying paper, equations, and expected behavior first.
3. **Never fabricate** experimental results, citations, metrics, baselines, or claims. If a number isn't measured or sourced, don't write it down as if it were.
4. Before implementing a research method, verify the source paper/methodology, equations, assumptions, and expected behavior against the actual reference (not memory).
5. When modifying existing code, understand the **complete pipeline** first and check for unintended side effects (e.g., a change to preprocessing silently affecting evaluation).
6. Clearly distinguish, in code comments/PRs/reports, between:
   - Established methods from existing literature
   - The user's proposed/novel methodology
   - Implementation assumptions made along the way
   - Experimental observations
7. Use appropriate baselines and controlled comparisons. Do not claim improvement without experimental evidence.
8. Ensure experiments are reproducible: fixed seeds, clearly defined configs, proper dataset splits, and recorded hyperparameters.
9. Use standard, task-appropriate evaluation metrics: mAP, precision, recall, robustness metrics, attack success rate (ASR), confidence degradation, localization failure, and other task-specific metrics as relevant.
10. Actively check for common research errors:
    - Data leakage
    - Incorrect preprocessing
    - Inconsistent evaluation settings across methods/baselines
    - Unfair baseline comparisons
    - Gradient mistakes
    - Incorrect attack implementations
    - Accidental use of test data during training
11. For adversarial attacks, verify: perturbation definitions, threat model, epsilon/step size, number of iterations, norm constraints (L∞/L2/etc.), targeted vs. untargeted setting, and the attack objective.
12. For defenses, verify evaluation against the **correct threat model**, and where applicable, against **adaptive or stronger attacks** (not just the attacks the defense was designed around).
13. Keep code modular, documented, testable, and suitable for eventual release/reproduction.
14. When multiple technically valid approaches exist, briefly explain the trade-offs and pick the one most appropriate for a rigorous research experiment.
15. **Do not silently guess.** If an important research detail is ambiguous or underspecified, stop and ask rather than implementing an unsupported assumption.
16. Before declaring a task complete, perform sanity checks and verify the implementation actually matches the intended methodology.

## Git Commits

- Never add a "Co-Authored-By" line or any co-author attribution to git commit messages or pull request descriptions.

## Notes for Working in This Repo

- `AntiStyler/` contains the reference implementation, demo notebook (`AntiStyler_Demo.ipynb`), and the paper transcription. Treat the paper (`AntiStyler_paper.md`) as the ground truth for the method's equations, pipeline phases, and reported numbers — cross-check any reimplementation or extension against it rather than reconstructing the method from memory.
- When adding new experiments, baselines, or attacks, follow the same evaluation protocol used in the paper (dataset splits, mAP@IoU=0.5, benign/adv/mean sub-columns, processing time measured as mean over all images) unless there's an explicit, stated reason to deviate — and if you deviate, say so explicitly rather than silently changing the protocol.

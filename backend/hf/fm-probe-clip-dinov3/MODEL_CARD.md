---
license: other
license_name: dinov3-license
license_link: LICENSE-DINOv3.md
library_name: timm
pipeline_tag: image-classification
tags:
  - ai-generated-image-detection
  - deepfake-detection
  - linear-probe
  - dinov3
  - clip
base_model:
  - timm/vit_base_patch16_dinov3.lvd1689m
  - timm/vit_base_patch16_clip_224.openai
---

# fm-probe-clip-dinov3: AI-generated image detector (frozen CLIP + DINOv3, linear probe)

A calibrated linear classifier on frozen features from two vision foundation models. It scores
how likely a still image is AI-generated and returns a three-way verdict: **ai**, **inconclusive**
or **real**. It is the production model of the DeepFake Detector app (MacAI Society).

This repo holds the **head only** (`head.npz`, `config.json`). The backbones are loaded from
their upstream timm repos at the pinned revisions in `config.json`:

| backbone | timm model | revision | licence |
|---|---|---|---|
| CLIP ViT-B/16 (OpenAI) | `timm/vit_base_patch16_clip_224.openai` | `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` | MIT (OpenAI CLIP); timm repo card: Apache-2.0 |
| DINOv3 ViT-B/16 (LVD-1689M) | `timm/vit_base_patch16_dinov3.lvd1689m` | `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da` | DINOv3 License (see `LICENSE-DINOv3.md`) |

## How it works

1. **Canonicalise**: EXIF transpose, RGB, short side to 512 px (Lanczos), JPEG q90 round trip,
   short side to 224 (bicubic), center crop 224, per-backbone timm mean/std.
2. **Features**: per backbone, the L2-normalised CLS token after blocks 6 and 9 and the
   L2-normalised pre-logits pooled token; both backbones concatenated (4,608 dims), fp16 inference.
3. **Head**: StandardScaler + LogisticRegression (C=0.1), then Platt scaling
   `p = sigmoid(a * clip(z, -13.8155, 13.8155) + b)` fitted on the validation split.
4. **Decision**: binary threshold 0.311 (Youden on validation) for `pred`; verdict bands
   `[0.2, 0.7]`: `real` below 0.2, `ai` at or above 0.7, `inconclusive` in between.
   On validation, 3.0% of fakes fall below 0.2 and 5.1% of reals reach 0.7; 13.7% of reals and
   12.7% of fakes are inconclusive.

`head.npz` is plain numpy (no pickle): `scaler_mean`, `scaler_scale`, `coef`, `intercept`,
`platt_a`, `platt_b`, `threshold`, `logit_clip`, `format_version`, `head_version`,
`feature_spec_hash`. The serving code refuses a head whose feature-spec hash does not match
`config.json`.

## Training data

All images canonicalised as above before feature extraction.

- OpenFake (old train split, ~20k images) minus the images used as reals in our manual test set.
- OpenFake core-train: 721 images from 45 recent generators, weighted to equal total weight with
  the old split.
- 400 COCO-train real photos for real-photo diversity.

Validation (Platt scaling, threshold, verdict bands, all hyper-parameter choices): OpenFake
core-val, canonicalised. It shares generators with core-train. Nothing was tuned on the test sets.

## Evaluation

AUROC on "shortcut-free" sets, where format and size cannot separate the classes (metadata-only
AUROC is 0.50); 95% bootstrap CIs in brackets.

| set | this model | previous 4-model fusion |
|---|---|---|
| core-test-sq (2025-26 generators, all unseen in training; n=2661) | **0.938** [0.929, 0.946] | 0.527 |
| reddit-test-sq (in the wild; n=923) | **0.835** [0.805, 0.866] | 0.471 |
| indep-sq (COCO-val reals + manually generated fakes; n=1041) | **0.944** [0.930, 0.956] | n/a |

On core-test-sq: balanced accuracy 0.85 (previously 0.55), TPR at 1% FPR 0.55 (previously 0.10),
ECE 0.088. Weakest large generator group: flux.2-klein-9b, AUROC 0.80. Robustness: JPEG q75
costs 0.02-0.03 AUROC, 0.5x downscaling at most 0.005. indep-sq is optimistic because the training
reals include COCO-train.

## Limitations

- **High-quality real photos are often flagged.** At the 0.311 threshold the false-positive rate is
  16-30% on DOCCI, Pexels and reddit real photos (1-3% on ImageNet/LAION). Professional,
  high-resolution or heavily edited photos are the main failure mode. Use the inconclusive band,
  and do not treat an "ai" verdict on its own as proof.
- **Calibration is poor in the wild**: ECE about 0.24 on reddit-test-sq (0.03 on validation).
  Probabilities are not reliable confidence statements outside the training distribution.
- Only the central square of the image is analysed (224 px after resizing). Edits or generated
  regions outside it are not seen; high-resolution detail is lost.
- New generators drift: the leave-one-family-out penalty was up to 0.035 AUROC.
- Still images only; no video, audio or face-swap-specific modelling.
- The heatmaps are gradient x activation saliency of the linear logit: they show which patches
  moved the score, not where an artefact "is".

## Intended use

Decision support for people checking whether an image might be AI-generated, with the verdict,
probability and per-backbone explanation shown together. Not for automated moderation, legal or
disciplinary decisions without human review.

**Prohibited use (DINOv3 License, section 1.b.v):** no use for activities subject to ITAR or end
uses prohibited by trade controls, including military or warfare purposes, nuclear applications,
espionage, or the development or use of guns or illegal weapons.

## Licence and attribution

- This model includes and depends on **DINOv3** ("DINO Materials") by Meta, used under the
  **DINOv3 License** (last updated August 19, 2025). The full text is in `LICENSE-DINOv3.md` and
  must be shipped with any redistribution of this model or derivatives. Use is subject to that
  licence, including its trade-control and acceptable-use terms.
- DINOv3: Siméoni et al., *DINOv3*, Meta AI, 2025. Publications using this model must acknowledge
  the use of DINOv3.
- CLIP: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*,
  ICML 2021 (OpenAI, MIT licence). Weights via timm (Ross Wightman, `huggingface/pytorch-image-models`).
- OpenFake dataset (ComplexDataLab) and COCO (Lin et al., 2014; CC BY 4.0 annotations, Flickr
  image terms) for training data.

## Versioning

`head_version` and `feature_spec_hash` in `head.npz` identify the head; the API reports them as
`model_version`. When the head is refit, re-run the export script (`backend/app/scripts/export_probe_head.py`)
and update the threshold, bands and metrics in this card.

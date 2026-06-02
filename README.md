# STS-Net

Phonological property prediction for Swedish Sign Language (STS) from pose keypoints.

**Current version: v0.2** — clip-level attention-pooled model (`ClipClassifier`).
For the v0.1 per-frame BiLSTM model see [README_v01.md](README_v01.md).

---

## What it predicts

Given a short sign clip, the model predicts seven phonological properties:

| Head | Type | Labels |
|------|------|--------|
| shape (dom.) | multi-hot BCE | 42 dominant handshapes |
| att (dom.) | multi-hot BCE | 34 orientation slugs |
| contact_loc | multi-hot BCE | 22 contact locations |
| contact_type | multi-hot BCE | 4 contact types |
| motion | multi-hot BCE | 8 motion directions (incl. none) |
| hand_type | CE | one / two |
| nondom_shape | multi-hot BCE | 42 non-dominant handshapes (optional head) |

Multi-hot targets allow signs with multiple phases (handform changes) to be
represented correctly. Top-1 accuracy is used as the evaluation metric.

## Architecture

Each clip is encoded through four MediaPipe pose streams (dominant hand 21 kpts,
non-dominant hand 21 kpts, upper body 12 kpts, face 25 kpts). Each stream passes
through a `FrameEncoder` (linear projection → 3 × temporal Conv1d, residual).
The four 256-dim outputs are fused by a linear MLP and then aggregated by
**masked attention pooling** — attention scores outside the annotated sign window
are forced to −∞, so the pooled 256-dim clip embedding represents only the core
signing portion. Seven classification heads operate on this embedding.

Optional additional streams: WiLoR 3D MANO joints (`wilor_dom`, `wilor_nondom`),
DINOv2 CLS tokens (`dino`), Moryossef rotation-normalised hands (`dom_norm`, `nondom_norm`),
or a Sapiens whole-body stream in place of MediaPipe.

Total parameters: ~4.3 M.

## Installation

```bash
git clone git@github.com:jbeskow/stsnet.git
cd stsnet
git lfs pull          # download v0.1 checkpoint (~213 MB) if needed
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, and [pose-format](https://github.com/sign-language-processing/pose-format).

---

## Python API

```python
from stsnet import ClipClassifierInference

model = ClipClassifierInference("checkpoints/stsnet_v02.pt", device="cuda")

# Predict phonological properties for a sign window
props = model.predict_phonology("clip.pose", sign_start=12, sign_end=58)
# {"shape": "Flata handen", "att": "vänsterriktad-nedåtvänd",
#  "cloc": "none", "ctype": "none", "motion": "none",
#  "hand_type": "one", "nondom_shape": "..."}

# 256-dim clip embedding for retrieval / clustering
emb = model.embed_clip("clip.pose", sign_start=12, sign_end=58)
# np.ndarray shape (256,)
```

---

## Training

### Prerequisites

- SSLL dataset: `sign_data_with_signer_fr.csv`, `.pose` files, `pseudo_signing.json`
  (see [README_v01.md](README_v01.md) for pose extraction and pseudo-signing steps)
- Pose cache built with `cache_poses.py` (see CLAUDE.md)

### Train on SSLL only

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/train_clip.py \
    --out runs/clip_v02
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--streams` | `dom nondom body face` | Input streams to use |
| `--hidden_dim` | 256 | Encoder channel width |
| `--epochs` | 60 | Training epochs |
| `--dropout` | 0.2 | Dropout rate |
| `--no_z` | off | Use 2D (xy) pose only |
| `--nondom_shape_head` | off | Enable non-dominant handshape head |
| `--ckpt` | None | Warm-start encoders from v0.1 checkpoint |
| `--wilor_dir` | None | Add WiLoR 3D hand streams |
| `--dino_dir` | None | Add DINOv2 CLS token stream |

Checkpoint saved to `runs/clip_v02/best.pt` (includes vocab metadata).

### Mine SSLC and re-train

Transfer phonological labels from SSLL to unannotated continuous-signing data
via nearest-neighbour embedding matching:

```bash
# Step 1: mine SSLC
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/mine_sslc.py \
    --clip_ckpt runs/clip_v02/best.pt \
    --threshold 0.4 \
    --out runs/sslc_mined.json

# Step 2: re-train with mined data + stronger regularisation
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/train_clip.py \
    --out runs/clip_v02_mined \
    --clip_ckpt runs/clip_v02/best.pt \
    --mined_json runs/sslc_mined.json \
    --dropout 0.3 --noise_std 0.02 --label_smoothing 0.1 \
    --time_stretch_min 0.85 --time_stretch_max 1.15
```

Mining selects SSLC gloss windows whose embedding (cosine distance) to the nearest
SSLL training variant is below `--threshold`. Mined instances carry the matched
SSLL phonological targets.

### Performance (MediaPipe baseline, SSLL val set)

| Property | SSLL only | +mined SSLC |
|----------|-----------|-------------|
| Handshape (dom.) | 79.9% | **85.1%** |
| Attitude | 75.9% | **79.3%** |
| Contact location | 78.3% | **80.6%** |
| Contact type | 72.3% | **76.8%** |
| Motion direction | 57.4% | **62.4%** |
| Hand type | 96.6% | **97.1%** |
| Handshape (nondom.) | 81.3% | **85.7%** |

---

## Repository layout

```
stsnet/                      package
  __init__.py                exports STSNet (v0.1), ClipClassifier (v0.2), both Inference APIs
  clip_classifier.py         ClipClassifier + AttentionPool  ← v0.2 default model
  model.py                   STSNet (per-frame BiLSTM)       ← v0.1
  encoder.py                 FrameEncoder, DinoEncoder, TemporalConvBlock
  inference.py               ClipClassifierInference (v0.2), STSNetInference (v0.1)
  train_utils.py             loss / accuracy helpers
  viterbi.py                 blank-free Viterbi forced alignment (v0.1)
  data/
    pose_io.py               MediaPipe loading, load_wilor_streams, normalize_hand_moryossef
    ssll_clip.py             SSLLClipDataset, collate_clip, load_sapiens_streams
    sslc_mined.py            SSLCMinedDataset (mined SSLC gloss clips)
    description.py           Swedish sign description parser
    contact.py               contact location / type vocabularies
    multihead.py             SSLLMultiHeadDataset (per-frame, used by v0.1)
    align_dataset.py         STSAlignDataset + emission builder (used by v0.1)
scripts/
  train_clip.py              train ClipClassifier v0.2       (stsnet-train-clip)
  mine_sslc.py               mine SSLC via embedding matching (stsnet-mine)
  train.py                   train STSNet v0.1               (stsnet-train)
  predict.py                 per-frame inference on .pose file
  align.py                   re-align a dataset
  evaluate.py                score alignment vs. manual annotations
  recipe.py                  full v0.1 cold-start recipe
  extract_pose.py            extract MediaPipe pose from mp4
  generate_pseudo_signing.py compute sign windows from pose
  make_seed_alignment.py     initial equal-split alignment
checkpoints/
  stsnet_base.pt             v0.1 pretrained checkpoint (Git LFS, ~213 MB)
data/
  sts_handformer.txt         handshape vocabulary (42 classes)
  annotations2.json          100-clip manual boundary annotations (test set)
  test_list2.json            test set clip list
  signer_map.csv             video_id → signer mapping
config/
  default.yaml               v0.1 training hyperparameters
tests/
  test_viterbi.py            unit tests for ctc_forced_align
```

---

## v0.1 model

The v0.1 per-frame BiLSTM model (`STSNet`) predicts nine phonological features
per frame and supports Viterbi forced alignment for timeline annotation.
See [README_v01.md](README_v01.md) for full documentation and the pretrained
checkpoint at `checkpoints/stsnet_base.pt`.

# PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis

> **PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis**
> Dinglun He\*, Baoming Zhang\*, Xu Wang\*, Yao Hao, Deshan Yang, Ye Duan
> (\* Equal contribution)
> *IEEE International Symposium on Biomedical Imaging (ISBI), 2026*
> [[Paper]](#) | [[Data](https://zenodo.org/records/10047292)]

---

## Overview

PIVM is a diffusion-based generative model for volumetric medical image synthesis. It learns to reconstruct CT image slices conditioned on organ segmentation masks and intensity-based label maps, using a residual learning strategy within a denoising diffusion probabilistic framework.

Key features:
- **Residual diffusion learning**: the model predicts the residual between a CT image and its organ-intensity label, rather than raw pixel values
- **Sequential volume generation**: slices are generated one by one, each conditioned on the previous generated slice
- **Multi-modal conditioning**: combines organ masks, intensity labels, and previous-slice context

---

## Requirements

```bash
pip install torch torchvision albumentations numpy opencv-python einops tqdm Pillow
```

Tested with Python 3.8+, PyTorch 2.0+, CUDA 11.8+.

---

## Data

> **Download:** [https://zenodo.org/records/10047292](https://zenodo.org/records/10047292)

After downloading, the raw dataset follows the TotalSegmentator format:

```
Totalsegmentator_dataset/
└── s0001/
    ├── ct.nii.gz
    └── segmentations/
        ├── liver.nii.gz
        ├── spleen.nii.gz
        ├── pancreas.nii.gz
        └── ...
```

---

## Data Preprocessing

All preprocessing scripts are located in [`preprocessing/`](preprocessing/). Run them **in order**.

### Step 1 — Organ segmentation with TotalSegmentator

Generate per-organ segmentation masks from raw CT volumes using [TotalSegmentator](https://github.com/wasserth/totalsegmentator):

```bash
pip install totalsegmentator

# Run for each subject
TotalSegmentator -i Totalsegmentator_dataset/s0001/ct.nii.gz \
                 -o Totalsegmentator_dataset/s0001/segmentations/
```

### Step 2 — Merge organ segmentations into a single mask

[`preprocessing/ts_process.py`](preprocessing/ts_process.py) combines individual per-organ NIfTI files into a single multi-label `mask.nii` per subject. The 14 abdominal organs (liver, spleen, kidneys, pancreas, stomach, lungs, gallbladder, adrenal glands) are each assigned a unique integer label (1–14).

```bash
python preprocessing/ts_process.py
```

> Edit `dataset_path` inside the script to point to your `Totalsegmentator_dataset_v201/` directory.

Output: `mask.nii` saved inside each subject folder.

### Step 3 — Convert NIfTI volumes to 2D PNG slices

[`preprocessing/vol2imglabel.py`](preprocessing/vol2imglabel.py) converts each subject's CT volume and merged mask into 2D PNG slices:
- Clips CT to **[-200, 500] HU**, normalizes to [0, 255]
- Pads to square, resizes to **256×256**
- Skips slices where the mask is empty
- Saves CT slices → `./image/` and mask slices → `./label/`

```bash
python preprocessing/vol2imglabel.py
```

> Edit `dataset_dir` inside the script to point to your dataset. Ensure `./image/` and `./label/` directories exist beforehand.

Output: `{subject}_{slice}.png` files in `./image/` and `./label/`.

### Step 4 — Generate intensity-based conditioning labels

[`preprocessing/process_ct_intensity.py`](preprocessing/process_ct_intensity.py) replaces each organ region in the mask with its **global mean CT intensity** across the full dataset, producing the conditioning label maps used during training.

```bash
python preprocessing/process_ct_intensity.py \
  --ct_dir    ./image \
  --label_dir ./label \
  --output_dir ./data/train/label
```

| Argument | Default | Description |
|---|---|---|
| `--ct_dir` | `./data/images` | Directory of CT slice PNGs (from Step 3) |
| `--label_dir` | `./data/labels` | Directory of organ mask PNGs (from Step 3) |
| `--output_dir` | `./data/output` | Where to save intensity label maps |

### Final directory structure

After all preprocessing steps, organize files as:

```
data/
├── train/
│   ├── image/       # CT slices (Step 3)
│   ├── label/       # Intensity label maps (Step 4)
│   └── organ/       # Organ mask slices (Step 3)
└── test/
    ├── image/
    ├── label/
    └── organ/
```

Files follow the naming convention: `{subject}_{slice_number}.png`

```
s0001_214_1.png   ← subject s0001, volume 214, slice 1
s0001_214_2.png   ← subject s0001, volume 214, slice 2
```

> Note: `_1.png` slices are used as reference and are excluded from training.

---

## Training

```bash
python train_test_ddpm.py \
  --image_dir  ./data/train/image \
  --label_dir  ./data/train/label \
  --organ_dir  ./data/train/organ \
  --device     cuda
```

Key training hyperparameters (can be passed as arguments):

| Argument | Default | Description |
|---|---|---|
| `--lr` | `2e-5` | Learning rate (AdamW) |
| `--batch_size` | `1` | Batch size |
| `--emb_dim` | `256` | Timestep embedding dimension |
| `--image_size` | `256` | Input image resolution |
| `--num_workers` | `4` | DataLoader workers |

To resume from a checkpoint:

```bash
python train_test_ddpm.py \
  --load_model \
  --checkpoint_path ./results/checkpoints/ddpm27.pth.tar \
  --image_dir ./data/train/image \
  --label_dir ./data/train/label \
  --organ_dir ./data/train/organ
```

**Training outputs** are saved to:
```
results/
├── checkpoints/       # Model checkpoints (saved when loss improves)
└── *.png              # Intermediate sample visualizations
```

---

## Testing

### Paired inference (single slices)

```bash
python train_test_ddpm.py \
  --load_model \
  --checkpoint_path ./results/checkpoints/ddpm27.pth.tar \
  --image_dir  ./data/test/image \
  --label_dir  ./data/test/label \
  --organ_dir  ./data/test/organ \
  --output_dir ./results \
  --device     cuda
```

Outputs are saved to:
```
results/
├── generated/    # Model predictions
├── image/        # Input CT images
├── label/        # Conditioning labels
└── noise/        # Predicted noise maps
```

### Sequential volume reconstruction

For slice-by-slice volumetric generation (each slice conditioned on the previous):

```bash
python train_test_ddpm.py \
  --load_model \
  --checkpoint_path ./results/checkpoints/ddpm27.pth.tar \
  --image_dir  ./data/test/image \
  --label_dir  ./data/test/label \
  --organ_dir  ./data/test/organ \
  --output_dir ./results \
  --device     cuda
```

To resume sequential generation from a specific case:

```bash
python train_test_ddpm.py \
  ... \
  --resume_from s0001_214
```

---

## Model Architecture

The model is a U-Net with Transformer blocks trained as a denoising diffusion probabilistic model (DDPM).

- **Input:** 3-channel concatenation of `[noisy_residual, previous_slice, organ_mask]`
- **Backbone:** U-Net encoder–decoder with skip connections
- **Attention:** Local Self-Attention (LSA) at each scale — multi-head attention with learnable temperature and masked self-interaction
- **Timestep conditioning:** Sinusoidal positional encoding injected at every block
- **Output:** Predicted noise on the residual image
- **Noise schedule:** Cosine annealing, 1000 steps
- **Loss:** MSE + L1 on predicted noise

---

## Acknowledgements

Organ segmentation masks are generated using [TotalSegmentator](https://github.com/wasserth/totalsegmentator). We thank the authors for making their tool publicly available.

---

## Citation

```bibtex
@article{pivm,
  title   = {PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis},
  author  = {He, Dinglun and Zhang, Baoming and Wang, Xu and Hao, Yao and Yang, Deshan and Duan, Ye},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

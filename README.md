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
conda env create -f environment.yml
conda activate pivm
```

Tested with Python 3.9, PyTorch 1.12.1, CUDA 11.3.

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

### Step 1 — Merge organ segmentations into a single mask

[`preprocessing/ts_process.py`](preprocessing/ts_process.py) combines individual per-organ NIfTI files into a single multi-label `mask.nii` per subject. 

```bash
python preprocessing/ts_process.py
```
### Step 2 — Convert NIfTI volumes to 2D PNG slices

[`preprocessing/vol2imglabel.py`](preprocessing/vol2imglabel.py) converts each subject's CT volume and merged mask into 2D PNG slices.

```bash
python preprocessing/vol2imglabel.py
```
### Step 3 — Generate intensity-based conditioning labels

[`preprocessing/process_ct_intensity.py`](preprocessing/process_ct_intensity.py) replaces each organ region in the mask with its **global mean CT intensity** across the full dataset, producing the conditioning label maps used during training.

```bash
python preprocessing/process_ct_intensity.py \
  --ct_dir    ./image \
  --label_dir ./label \
  --output_dir ./data/train/label
```
---

## Training

Set `training = True` at the top of `train_test_ddpm.py`, configure the paths inside the script, then run:

```bash
python train_test_ddpm.py
```

To resume from a checkpoint, set `load_model = True` and point `checkpoint_path` to your saved checkpoint.

**Training outputs** are saved to:
```
results/
├── checkpoints/       # Model checkpoints (saved when loss improves)
└── *.png              # Intermediate sample visualizations
```

---

## Testing

Set `training = False` at the top of `train_test_ddpm.py`, configure the paths inside the script, then run:

```bash
python train_test_ddpm.py
```

This runs **sequential volume reconstruction** — slices are generated one by one, each conditioned on the previous generated slice. Results are saved to `results/`.

To resume from a specific case:

```python
test_args.resume_from = "s0001_214"  # set inside the script
```

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

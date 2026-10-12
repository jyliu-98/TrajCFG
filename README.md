# TrajCFG: Token-Wise Classifier-Free Guidance via Denoising Trajectory Feedback

<p align="center">
  Official PyTorch implementation of <strong>TrajCFG</strong>.
</p>

<p align="center">
  <a href="https://jyliu-98.github.io/trajcfg-project-page/"><img src="https://img.shields.io/badge/Project-Page-69b3a2" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv coming soon">
  <a href="https://github.com/jyliu-98/TrajCFG"><img src="https://img.shields.io/github/stars/jyliu-98/TrajCFG?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=u7Dqok8AAAAJ">Jingyu Liu</a><sup>1,†,§</sup>&emsp;
  Hao Jiang<sup>2,†</sup>&emsp;
  Yikang Ding<sup>2,‡</sup>&emsp;
  Huafeng Shi<sup>2</sup>&emsp;
  Xu Yang<sup>2</sup><br>
  Xiaoqiang Liu<sup>2</sup>&emsp;
  Guoxin Zhang<sup>2</sup>&emsp;
  Pengfei Wan<sup>2</sup>&emsp;
  Kun Gai<sup>2,*</sup>
</p>

<p align="center">
  <sup>1</sup>Renmin University of China&emsp;&emsp;
  <sup>2</sup>Kling Team, Kuaishou Technology
</p>

<p align="center">
  <sup>†</sup>Equal contribution&emsp;
  <sup>‡</sup>Project Leader&emsp;
  <sup>*</sup>Corresponding author&emsp;
  <sup>§</sup>Work done at Kling Team, Kuaishou Technology
</p>

<p align="center">
  <a href="https://jyliu-98.github.io/trajcfg-project-page/">
    <img src="https://jyliu-98.github.io/trajcfg-project-page/assets/images/method-overview.png" width="88%" alt="TrajCFG method overview">
  </a>
</p>

## Overview

Classifier-free guidance typically applies one shared scale to all latent
tokens, even though their velocity predictions evolve differently during
sampling. **TrajCFG** turns denoising history into a fine-grained control
signal: tokens with larger directional changes receive stronger guidance,
while tokens following more consistent trajectories receive weaker guidance.

TrajCFG is:

- **Training-free** — no fine-tuning or auxiliary model is required.
- **Token-wise** — each latent space-time location receives its own CFG scale.
- **Compute-neutral** — it reuses standard conditional and unconditional
  predictions with no additional model evaluations.
- **Modality-agnostic** — evaluated on text-to-video, image-to-video,
  text-to-audio-video, text-to-image, and image-to-3D generation.

At denoising step `k`, TrajCFG compares historical guided velocities and maps
their token-wise cosine similarity to bounded guidance scales:

```text
q[k,p] = cosine(v_hat[k-2,p], v_hat[k-1,p])
s[k,p] = s_min + (s_max - s_min) * clip((1 - q[k,p]) / (1 - rho_lo), 0, 1)
```

From step 3 onward, adjacent-step feedback is fused with interval-normalized
stride-2 feedback using weights `0.8/0.2`.

## Installation

```bash
git clone https://github.com/jyliu-98/TrajCFG.git
cd TrajCFG
```

### Wan backbones

Please follow the official installation instructions for
[Diffusers](https://github.com/huggingface/diffusers) and the corresponding
[Wan2.1](https://github.com/Wan-Video/Wan2.1) or
[Wan2.2](https://github.com/Wan-Video/Wan2.2) backbone.

### Cosmos-Predict2.5

Please follow the official installation instructions for
[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) and run
the Cosmos example in its environment.

## Usage

Every backbone is implemented as a self-contained script containing the full
TrajCFG feedback rule, denoising loop, model loading, and output handling.

```bash
# Text-to-video
python examples/t2v/wan21_1_3b.py
python examples/t2v/wan22_5b.py
python examples/t2v/wan22_14b.py

# Image-to-video
python examples/i2v/wan22_5b.py
python examples/i2v/wan22_14b.py

# Image-to-video with Cosmos-Predict2.5
python examples/i2v/cosmos_predict2_5_2b.py \
  --checkpoint-path /path/to/Cosmos-Predict2.5-2B
```

Prompts, seeds, model paths, output paths, and TrajCFG hyperparameters can be
overridden through command-line arguments. Outputs are written to `outputs/`;
each MP4 is accompanied by a JSON file containing the generation settings and
per-step scale statistics.

One non-VBench qualitative example is provided for each task/backbone. See
[assets/README.md](assets/README.md) for the corresponding inputs and results.

## Reproduced settings

| Task | Backbone | `s_base` | `s_min` | `s_max` | `rho_lo` | Steps |
|:---:|---|---:|---:|---:|---:|---:|
| T2V | Wan2.1-T2V-1.3B | 6.0 | 5.0 | 10.0 | 0.95 | 40 |
| T2V | Wan2.2-TI2V-5B | 5.0 | 3.0 | 10.0 | 0.95 | 40 |
| T2V | Wan2.2-T2V-A14B | 3.5 | 1.0 | 10.0 | 0.90 | 40 |
| I2V | Wan2.2-TI2V-5B | 5.0 | 2.0 | 10.0 | 0.95 | 40 |
| I2V | Wan2.2-I2V-A14B | 2.5 | 1.0 | 7.0 | 0.95 | 40 |
| I2V | Cosmos-Predict2.5-2B | 5.0 | 4.0 | 9.0 | 0.90 | 35 |

These defaults match Table 7 of the paper. All examples use adjacent/stride-2
fusion weights `(0.8, 0.2)`.

## Repository structure

```text
TrajCFG/
├── examples/
│   ├── t2v/                 # Self-contained text-to-video examples
│   └── i2v/                 # Self-contained image-to-video examples
├── assets/README.md         # Qualitative example inputs and results
└── README.md
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{liu2026trajcfg,
  title   = {TrajCFG: Token-Wise Classifier-Free Guidance via Denoising Trajectory Feedback},
  author  = {Liu, Jingyu and Jiang, Hao and Ding, Yikang and Shi, Huafeng and
             Yang, Xu and Liu, Xiaoqiang and Zhang, Guoxin and Wan, Pengfei and Gai, Kun},
  journal = {arXiv preprint},
  year    = {2026}
}
```

The arXiv identifier will be added after the preprint is released.

## Acknowledgements

This repository builds upon the following open-source projects:

- [Diffusers](https://github.com/huggingface/diffusers), including its
  [Wan pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan).
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) and
  [Wan2.2](https://github.com/Wan-Video/Wan2.2).
- [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5).

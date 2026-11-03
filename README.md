# Sampling with VAE-GAN

A lightweight inference pipeline that uses a pretrained VAE-GAN model to generate 10 diverse facial variations from a single input face image.

---

## 1. Project Overview

The objective of this project is to encode a single input facial image into latent space, apply controlled latent sampling, and decode the representations to generate exactly 10 distinct, domain-consistent facial variations alongside the input reconstruction:

$$\text{Input Face Image} \longrightarrow \text{Pretrained Encoder} \longrightarrow \text{Latent Sampling} \longrightarrow \text{Decoder} \longrightarrow \text{10 Generated Faces}$$

This repository focuses purely on inference and controlled latent-space exploration; no training or fine-tuning is required.

---

## 2. Features

- **Robust Input Preprocessing**: Supports JPG, JPEG, and PNG formats across RGB, RGBA, and grayscale. Center-crops without aspect ratio distortion and normalizes images to $128 \times 128 \times 3$.
- **Controlled Latent Sampling**: Uses posterior-guided latent sampling with the model's learned mean ($z_\mu$) and standard deviation ($z_\sigma = \exp(z_{\log \sigma})$) scaled by sampling factor $\alpha$.
- **High-Diversity Generation**: Independent Gaussian noise per sample ensures 10 distinct facial variations across expressions, hairstyles, lighting, and facial features.
- **Accurate Output Processing**: Normalizes decoder output into standard $[0, 1]$ RGB, saves 10 individual variation images, the direct reconstruction, and consolidated visualization grids.
- **Modern CLI**: Standardized command-line interface with arguments for input/output paths, sample count, checkpoint path, variation strength, and random seed.
- **Framework Compatibility**: Supports TensorFlow 1.15 as well as modern TensorFlow 2.x environments via `compat.v1`.

---

## 3. Model Architecture & Latent Sampling Method

### Architecture
The model uses the VAE-GAN architecture introduced by VanRullen & Reddy (2019):
- **Encoder**: 5-layer convolutional neural network (channels: 192, 256, 384, 512, 768; stride 2; kernel $3 \times 3$) with batch normalization and ELU activations, followed by a dense layer mapping to 1024 hidden units.
- **Variational Bottleneck**: Two linear heads compute the latent mean $z_\mu \in [-1, 1]$ and log standard deviation $z_{\log \sigma} \in [-0.5, 0.5]$ via $\tanh$ activations.
- **Decoder / Generator**: A dense projection layer expands $z$ to spatial feature maps ($4 \times 4 \times 1024$), followed by 5 transposed convolution layers (channels: 512, 384, 256, 192, 3) with ELU activations to reconstruct a $128 \times 128 \times 3$ face.

### Latent Sampling Method
The latent vector for each sample $i$ is generated using the reparameterized posterior-guided formulation:

$$z_{\text{sample}} = z_\mu + \alpha \cdot z_\sigma \odot \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, I_{1024})$$

where:
- $z_\mu$ is the deterministic latent embedding of the input face.
- $z_\sigma = \exp(z_{\log \sigma})$ is the learned posterior standard deviation.
- $\epsilon_i$ is an independent 1024-dimensional standard Gaussian noise vector for sample $i$.
- $\alpha$ is the user-controlled sampling strength (`--variation_strength`).

---

## 4. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Make sure you have Python installed and that the pretrained model checkpoint (`vaegan_celeba.ckpt`) is available in the project root directory before running the pipeline.


### Checkpoint Setup

The pretrained model checkpoint `vaegan_celeba.ckpt` (~554 MB) is required for inference:

- Download automatically using the included script:
  ```bash
  python download_checkpoint.py
  ```
- Or place `vaegan_celeba.ckpt` directly in the project root.
- SHA-256 Checksum: `f97e76a9302481d45dca45a523904c91a94c8e39b94922c0f949df02e5c4948b`

---

## 5. How to Run

### Basic Inference Command
```bash
python VAEGAN_sampling.py --input example.jpg --num_samples 10 --variation_strength 1.0 --output Results/example/
```

### CLI Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *Required* | Path to input face image (JPG, JPEG, PNG) |
| `--output` | `-o` | `Results/<name>/` | Directory to save generated images and grids |
| `--num_samples` | `-n` | `10` | Number of facial variations to generate |
| `--checkpoint` | `-c` | `vaegan_celeba.ckpt` | Path to pretrained model checkpoint |
| `--crop_factor` | | `0.8` | Face center crop ratio (0.1 to 1.0) |
| `--variation_strength` | | `1.0` | Sampling perturbation scale $\alpha$ (controls diversity) |
| `--seed` | | `None` | Random seed for reproducible variations |

---

## 6. Controlled Experiments & Diversity Evaluation

To systematically evaluate sampling diversity and image quality, four experiments were conducted across variation strengths $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ using `--seed 42`:

### Reproduction Commands
```bash
# Experiment A: alpha = 0.5 (Subtle variation / low diversity)
python VAEGAN_sampling.py --input example.jpg --variation_strength 0.5 --seed 42 --output Results/alpha_05/

# Experiment B: alpha = 1.0 (Optimal recommended default)
python VAEGAN_sampling.py --input example.jpg --variation_strength 1.0 --seed 42 --output Results/alpha_10/

# Experiment C: alpha = 1.5 (High diversity / expressive changes)
python VAEGAN_sampling.py --input example.jpg --variation_strength 1.5 --seed 42 --output Results/alpha_15/

# Experiment D: alpha = 2.0 (Extreme diversity / artifact risk)
python VAEGAN_sampling.py --input example.jpg --variation_strength 2.0 --seed 42 --output Results/alpha_20/

# Run quantitative evaluation across all experiments:
python evaluate_diversity.py
```

### Experimental Results Comparison

| Experiment | Alpha ($\alpha$) | Pairwise SSIM $\downarrow$ | Pairwise MAE $\uparrow$ | Pretrained Latent Dist $\uparrow$ | Recon SSIM $\uparrow$ | Visual Quality & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp A** | `0.5` | $0.9485 \pm 0.0071$ | $0.0237 \pm 0.0038$ | $7.53$ | $0.9514$ | Near-identical faces; insufficient diversity. |
| **Exp B** | `1.0` | $0.8686 \pm 0.0122$ | $0.0406 \pm 0.0047$ | $13.44$ | $0.9062$ | **Recommended**. Clear facial variations with high quality. |
| **Exp C** | `1.5` | $0.7578 \pm 0.0224$ | $0.0611 \pm 0.0081$ | $19.38$ | $0.8326$ | High diversity; bold hairstyle and expression shifts. |
| **Exp D** | `2.0` | $0.6339 \pm 0.0348$ | $0.0824 \pm 0.0101$ | $24.55$ | $0.7390$ | Distortions, color shifts, and boundary blur emerge. |

### Selected Configuration: $\alpha = 1.0$
- **Evidence**: At $\alpha = 1.0$, pairwise SSIM drops significantly from $0.9485$ to $0.8686$, and latent space separation nearly doubles from $7.53$ to $13.44$. Facial features (smile, gaze, hairstyle, bone structure) show clear, distinguishable diversity while retaining high reconstruction fidelity ($0.9062$) and zero severe artifacts.
- For complete details, see [`SAMPLING_DIVERSITY_REPORT.md`](SAMPLING_DIVERSITY_REPORT.md).

---

## 7. Project Structure

```
├── VAEGAN_sampling.py            # Main inference CLI (encode → sample → decode)
├── evaluate_diversity.py         # Quantitative diversity evaluation script (SSIM, MAE, Latent)
├── make_network.py               # VAE-GAN computational graph architecture
├── utils.py                      # Image preprocessing, deprocessing, and layer ops
├── download_checkpoint.py        # Resilient Git LFS checkpoint downloader
├── test_sampling_pipeline.py     # Unit test suite
├── requirements.txt              # Pinned Python dependencies (numpy<2.0.0)
├── SAMPLING_DIVERSITY_REPORT.md  # Detailed experiment report on sampling diversity
├── example.jpg                   # Example input face image
├── vaegan_celeba.ckpt            # Pretrained CelebA model checkpoint (~554 MB)
├── Results/
│   ├── alpha_05/                 # Experiment A (alpha=0.5) outputs and grids
│   ├── alpha_10/                 # Experiment B (alpha=1.0) outputs and grids [Default]
│   ├── alpha_15/                 # Experiment C (alpha=1.5) outputs and grids
│   └── alpha_20/                 # Experiment D (alpha=2.0) outputs and grids
└── README.md                     # Project documentation
```

---

## 8. Credits and Original Repository

- **Author**: Mahmoud Shaheen | [LinkedIn](https://www.linkedin.com/in/mahmoud-a-shaheen-5b4b70251/)
- **Original Model & Weights**: Rufin VanRullen & Ramesh Reddy (2019), *"Reconstructing Faces from fMRI Patterns using Deep Generative Neural Networks"*, [rufinv/VAE-GAN-CelebA](https://github.com/rufinv/VAE-GAN-CelebA).
- **Architecture Reference**: Parag K. Mital, *Creative Applications of Deep Learning*.

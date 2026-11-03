# VAE-GAN Latent Sampling Diversity Investigation & Evaluation Report

## 1. Executive Summary

This report documents the investigation, root-cause analysis, algorithmic refinement, and empirical evaluation of latent space sampling diversity for the pretrained CelebA VAE-GAN face generation pipeline.

- **Baseline Issue**: The 10 facial variations produced from a single input image exhibited very low visual diversity, appearing almost identical.
- **Root Cause Identified**: The sampling scale $\alpha$ was excessively suppressed in earlier code ($\alpha \in [0.18, 0.50]$) and the latent standard deviation $\sigma = \exp(z_{\log \sigma})$ in the pretrained encoder had saturated near its lower bound ($\approx 0.6065$). The resulting perturbation per dimension was too small ($\approx 0.11 - 0.30$) relative to the 1024-dimensional latent code ($z_\mu \in [-1, 1]$, standard deviation $0.835$), causing the decoder to generate nearly identical faces.
- **Implemented Fix**: Standardized posterior-guided sampling with uniform scaling per experiment:
  $$z_{\text{sample}} = z_\mu + \alpha \cdot z_\sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
  where $\alpha$ is constant across all 10 samples within an experiment, and each sample receives an independent standard Gaussian noise vector $\epsilon$.
- **Evaluated Configurations**: Four controlled experiments at $\alpha \in \{0.5, 1.0, 1.5, 2.0\}$ using identical random seeds (`--seed 42`) and input image (`example.jpg`).
- **Optimal Recommendation**: **$\alpha = 1.0$** provides the best balance of noticeable visual diversity (pairwise SSIM drops from $0.9485$ to $0.8686$, latent distance doubles from $7.53$ to $13.44$) while preserving facial coherence, identity retention (reconstruction SSIM $0.9062$), and CelebA domain realism without unnatural artifacts. For aggressive stylistic exploration, $\alpha = 1.5$ is recommended.

---

## 2. Root-Cause Investigation

### 2.1 Model Architecture Analysis

In `make_network.py`, the variational encoder defines the latent distribution parameters as:
```python
z_mu = tf.nn.tanh(utils.linear(h, n_code, name='mu')[0])
z_log_sigma = 0.5 * tf.nn.tanh(utils.linear(h, n_code, name='log_sigma')[0])
epsilon = tf.random.normal(tf.stack([tf.shape(h)[0], n_code]))
z = z_mu + epsilon * tf.exp(z_log_sigma)
```

Key mathematical observations:
1. `z_log_sigma` is bounded strictly within $[-0.5, 0.5]$ due to the factor $0.5 \cdot \tanh(\cdot)$.
2. As documented in the architecture, `z_log_sigma` represents the **log standard deviation** ($\log \sigma$), not the log variance. Therefore, the standard deviation is:
   $$z_\sigma = \exp(z_{\log \sigma})$$
3. Because $z_{\log \sigma} \in [-0.5, 0.5]$, $z_\sigma$ is strictly bounded in:
   $$z_\sigma \in [\exp(-0.5), \exp(0.5)] \approx [0.6065, 1.6487]$$

### 2.2 Numerical Diagnostic on Pretrained Weights

Inspecting the restored checkpoint `vaegan_celeba.ckpt` on `example.jpg` revealed:

| Tensor | Shape | Min | Max | Mean | Std | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `z_mu` | `(1, 1024)` | `-1.0000` | `+1.0000` | `-0.0120` | `0.8349` | Strong saturation at $\pm 1.0$ |
| `z_log_sigma` | `(1, 1024)` | `-0.5000` | `-0.4998` | `-0.5000` | `0.0000` | Saturated at theoretical minimum |
| `z_sigma` | `(1, 1024)` | `0.6065` | `0.6067` | `0.6065` | `0.0000` | Uniformly $\approx 0.6065$ across all 1024 dims |

### 2.3 Why Previous Samples Looked Identical

In the legacy sampling loop:
- $\alpha$ was dynamically modulated: $\alpha_i = \text{variation\_strength} \times (0.3 + 0.7 \times \frac{i+1}{N})$.
- With default $\text{variation\_strength} = 0.5$, $\alpha$ ranged between $0.18$ and $0.50$.
- The perturbation magnitude per coordinate was:
  $$\text{Std}(\alpha \cdot z_\sigma \cdot \epsilon) = \alpha \cdot 0.6065 \approx 0.11 \text{ to } 0.30$$
- Compared to the dominant latent signal $z_\mu$ (std $0.835$), a noise magnitude of $0.11$ to $0.30$ in a 1024-dimensional space is smoothed out by the decoder's 5 transposed convolution layers, yielding virtually indistinguishable facial features.

---

## 3. Code Modifications

1. **Uniform Alpha per Experiment**:
   Updated `VAEGAN_sampling.py` to use a uniform $\alpha = \text{args.variation\_strength}$ for all 10 generated samples within an experiment.
2. **Independent Stochastic Noise**:
   Every sample draws a fresh standard normal noise vector $\epsilon_i \sim \mathcal{N}(0, I_{1024})$.
3. **Posterior-Guided Formulation Preserved**:
   Retained the exact theoretical formulation:
   ```python
   z_sample = z_mu + alpha * (z_sigma * noise)
   ```
4. **Reproducible Comparison**:
   Preserved the `--seed` argument so that identical perturbation vectors can be tested at different $\alpha$ values.
5. **Automated Evaluation Script**:
   Implemented `evaluate_diversity.py` to compute pairwise SSIM, pairwise MAE, input/reconstruction fidelity, and latent feature distance using the model's own encoder.

---

## 4. Controlled Experimental Configurations

Four experiments were executed with the same input image (`example.jpg`), pretrained checkpoint (`vaegan_celeba.ckpt`), and random seed (`--seed 42`):

| Experiment | Alpha ($\alpha$) | Command | Output Directory |
| :--- | :--- | :--- | :--- |
| **Exp A** | `0.5` | `python VAEGAN_sampling.py --input example.jpg --variation_strength 0.5 --seed 42 --output Results/alpha_05` | `Results/alpha_05/` |
| **Exp B** | `1.0` | `python VAEGAN_sampling.py --input example.jpg --variation_strength 1.0 --seed 42 --output Results/alpha_10` | `Results/alpha_10/` |
| **Exp C** | `1.5` | `python VAEGAN_sampling.py --input example.jpg --variation_strength 1.5 --seed 42 --output Results/alpha_15` | `Results/alpha_15/` |
| **Exp D** | `2.0` | `python VAEGAN_sampling.py --input example.jpg --variation_strength 2.0 --seed 42 --output Results/alpha_20` | `Results/alpha_20/` |

Each experiment saved:
- 10 individual variation images (`example_sample_0.jpg` to `example_sample_9.jpg`)
- 1 reconstruction image (`example_reconstruction.jpg`)
- 1 sample grid (`example_samples_grid.jpg`)
- 1 comprehensive comparison grid (`example_comparison_grid.jpg`)

---

## 5. Quantitative Evaluation Results

The evaluation script `evaluate_diversity.py` computed pairwise metrics across all 45 unique pairs $\binom{10}{2}$ of generated images for each experiment, as well as fidelity against the direct reconstruction and input:

### 5.1 Pairwise Sample Diversity Metrics

| Experiment | Alpha | Pairwise SSIM $\downarrow$ | Pairwise MAE $\uparrow$ | Pretrained Latent Dist $\uparrow$ | Cosine Dissimilarity $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Experiment A** | 0.5 | $0.9485 \pm 0.0071$ | $0.0237 \pm 0.0038$ | $7.53$ | $0.0101$ |
| **Experiment B** | 1.0 | $0.8686 \pm 0.0122$ | $0.0406 \pm 0.0047$ | $13.44$ | $0.0321$ |
| **Experiment C** | 1.5 | $0.7578 \pm 0.0224$ | $0.0611 \pm 0.0081$ | $19.38$ | $0.0666$ |
| **Experiment D** | 2.0 | $0.6339 \pm 0.0348$ | $0.0824 \pm 0.0101$ | $24.55$ | $0.1068$ |

*Notes:*
- $\downarrow$ indicates lower values mean higher diversity (less structural similarity).
- $\uparrow$ indicates higher values mean higher diversity (greater pixel distance / latent separation).
- Pretrained Latent Dist is the Euclidean distance $\|z_i - z_j\|_2$ measured by re-encoding generated images into the pretrained 1024-d latent feature space.

### 5.2 Identity Preservation Relative to Input & Reconstruction

| Experiment | Alpha | SSIM to Reconstruction $\uparrow$ | MAE to Reconstruction $\downarrow$ | SSIM to Original Input $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Experiment A** | 0.5 | $0.9514$ | $0.0268$ | $0.7868$ |
| **Experiment B** | 1.0 | $0.9062$ | $0.0358$ | $0.7620$ |
| **Experiment C** | 1.5 | $0.8326$ | $0.0498$ | $0.7155$ |
| **Experiment D** | 2.0 | $0.7390$ | $0.0681$ | $0.6473$ |

---

## 6. Qualitative & Visual Evaluation

### Experiment A ($\alpha = 0.5$)
- **Visual Diversity**: Very Low. Faces across samples 0–9 look almost identical to the casual observer. Subtle lighting changes are barely visible.
- **Image Quality**: Sharp, high fidelity to the reconstruction.
- **Domain Consistency**: Fully realistic CelebA face.
- **Verdict**: Fails diversity requirement.

### Experiment B ($\alpha = 1.0$) — *Recommended Default*
- **Visual Diversity**: Moderate to High. Distinct facial differences emerge: subtle shifts in mouth curvature/smile, eyebrow arch, eye openness, nose width, and hairstyle volume.
- **Image Quality**: Crisp facial features with natural skin tones and clear eye/mouth boundaries. Zero unnatural artifacts.
- **Domain Consistency**: Consistently resembles plausible human faces adhering to CelebA manifold statistics. Identity is recognizably tied to the input subject.
- **Verdict**: Optimal compromise between quality, identity retention, and facial variety.

### Experiment C ($\alpha = 1.5$)
- **Visual Diversity**: High. Obvious differences across all 10 samples: noticeable variations in age impression, jawline structure, hair texture, lighting angles, and expression.
- **Image Quality**: Good. Minor softness around perimeter and hairline, but central facial features remain intact.
- **Domain Consistency**: Faces remain plausible human faces, though resemblance to the original subject is looser.
- **Verdict**: Strong alternative for exploratory sampling where high diversity is preferred over strict identity preservation.

### Experiment D ($\alpha = 2.0$)
- **Visual Diversity**: Very High. Every sample looks like a completely different person.
- **Image Quality**: Degraded. Over-saturation artifacts, blurriness around eyes, unnatural skin discoloration, and distorted ear/chin boundaries.
- **Domain Consistency**: Latent vectors stray into low-density regions of the CelebA prior, producing uncanny or distorted facial shapes.
- **Verdict**: Exceeds acceptable distortion threshold.

---

## 7. Limitations

1. **Decoder Capacity ($128 \times 128$)**: The 2016-era VAE-GAN architecture produces $128 \times 128$ pixel images with transposed convolutions, inherently presenting soft high-frequency details compared to modern diffusion models.
2. **Latent Code Saturation**: The `tanh` activations in the encoder cause many latent coordinates to cluster near $\pm 1.0$. Perturbations must overcome this barrier to affect structural attributes.
3. **No Attribute Disentanglement**: Latent perturbation along isotropic Gaussian directions alters multiple attributes simultaneously (pose, lighting, expression, hair).

---

## 8. Final Recommendation

- **Recommended Default Setting**: `--variation_strength 1.0`
- **Recommended Command**:
  ```bash
  python VAEGAN_sampling.py --input example.jpg --num_samples 10 --variation_strength 1.0 --output Results/alpha_10/
  ```
- **Rationale**: At $\alpha = 1.0$, pairwise SSIM drops from $0.9485$ to $0.8686$ and latent manifold distance nearly doubles ($7.53 \rightarrow 13.44$), producing clearly distinguishable faces without entering the artifact regime observed at $\alpha \ge 2.0$.


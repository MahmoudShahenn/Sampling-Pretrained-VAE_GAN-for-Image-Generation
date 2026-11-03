#!/usr/bin/env python3
"""Evaluate Latent Sampling Diversity for VAE-GAN Face Generation.

Computes quantitative metrics across sampling experiments:
1. Pairwise Structural Similarity Index (SSIM) across samples (lower = more diverse).
2. Pairwise Mean Absolute Error (MAE / L1 pixel distance) (higher = more diverse).
3. Identity preservation: SSIM and MAE against input and direct reconstruction.
4. Latent feature space diversity: Pairwise Euclidean & Cosine distance in the
   pretrained VAE-GAN 1024-dimensional latent embedding space.
"""

import os
import glob
import itertools
import numpy as np
from PIL import Image
import scipy.ndimage

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
except (ImportError, AttributeError):
    try:
        import tensorflow as tf
    except ImportError:
        tf = None

import utils
import make_network


def ssim_channel(x, y, C1=0.01**2, C2=0.03**2):
    """Compute SSIM for a single color channel using uniform 11x11 window."""
    mu_x = scipy.ndimage.uniform_filter(x, size=11)
    mu_y = scipy.ndimage.uniform_filter(y, size=11)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = scipy.ndimage.uniform_filter(x * x, size=11) - mu_x_sq
    sigma_y_sq = scipy.ndimage.uniform_filter(y * y, size=11) - mu_y_sq
    sigma_xy = scipy.ndimage.uniform_filter(x * y, size=11) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    return np.mean(num / np.maximum(den, 1e-8))


def compute_ssim(img1, img2):
    """Compute mean 3-channel SSIM between two [0, 1] RGB images."""
    return float(np.mean([ssim_channel(img1[..., c], img2[..., c]) for c in range(3)]))


def compute_mae(img1, img2):
    """Compute Mean Absolute Error (MAE / L1 distance) between two images."""
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


def load_image_normalized(path):
    """Load image as float32 RGB in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0


def evaluate_experiment_directory(exp_dir, input_path="example.jpg", encoder_fn=None):
    """Evaluate diversity and fidelity metrics for an experiment folder."""
    sample_paths = sorted(glob.glob(os.path.join(exp_dir, "*_sample_*.jpg")))
    recon_paths = glob.glob(os.path.join(exp_dir, "*_reconstruction.jpg"))

    if not sample_paths:
        raise FileNotFoundError(f"No sample images found in {exp_dir}")

    samples = [load_image_normalized(p) for p in sample_paths]
    num_samples = len(samples)

    # Load input and reconstruction if available
    input_img = utils.preprocess128(input_path) if os.path.isfile(input_path) else None
    recon_img = load_image_normalized(recon_paths[0]) if recon_paths else None

    # 1. Pairwise diversity metrics across generated samples (45 pairs for 10 samples)
    pairwise_ssim = []
    pairwise_mae = []
    for (i, j) in itertools.combinations(range(num_samples), 2):
        pairwise_ssim.append(compute_ssim(samples[i], samples[j]))
        pairwise_mae.append(compute_mae(samples[i], samples[j]))

    # 2. Identity preservation against input & reconstruction
    recon_ssim = [compute_ssim(s, recon_img) for s in samples] if recon_img is not None else []
    recon_mae = [compute_mae(s, recon_img) for s in samples] if recon_img is not None else []
    input_ssim = [compute_ssim(s, input_img) for s in samples] if input_img is not None else []
    input_mae = [compute_mae(s, input_img) for s in samples] if input_img is not None else []

    # 3. Latent feature space diversity using pretrained encoder
    latent_pairwise_dist = []
    latent_cosine_dist = []
    if encoder_fn is not None:
        batch = np.stack(samples, axis=0)  # (N, 128, 128, 3)
        latents = encoder_fn(batch)  # (N, 1024)
        for (i, j) in itertools.combinations(range(num_samples), 2):
            z_i = latents[i]
            z_j = latents[j]
            euc_dist = np.linalg.norm(z_i - z_j)
            cos_sim = np.dot(z_i, z_j) / (np.linalg.norm(z_i) * np.linalg.norm(z_j) + 1e-8)
            latent_pairwise_dist.append(float(euc_dist))
            latent_cosine_dist.append(float(1.0 - cos_sim))

    return {
        "exp_dir": exp_dir,
        "num_samples": num_samples,
        "pairwise_ssim_mean": float(np.mean(pairwise_ssim)),
        "pairwise_ssim_std": float(np.std(pairwise_ssim)),
        "pairwise_mae_mean": float(np.mean(pairwise_mae)),
        "pairwise_mae_std": float(np.std(pairwise_mae)),
        "recon_ssim_mean": float(np.mean(recon_ssim)) if recon_ssim else None,
        "recon_mae_mean": float(np.mean(recon_mae)) if recon_mae else None,
        "input_ssim_mean": float(np.mean(input_ssim)) if input_ssim else None,
        "input_mae_mean": float(np.mean(input_mae)) if input_mae else None,
        "latent_euc_dist_mean": float(np.mean(latent_pairwise_dist)) if latent_pairwise_dist else None,
        "latent_cos_dist_mean": float(np.mean(latent_cosine_dist)) if latent_cosine_dist else None,
    }


def main():
    experiments = [
        ("Experiment A (alpha=0.5)", "Results/alpha_05", 0.5),
        ("Experiment B (alpha=1.0)", "Results/alpha_10", 1.0),
        ("Experiment C (alpha=1.5)", "Results/alpha_15", 1.5),
        ("Experiment D (alpha=2.0)", "Results/alpha_20", 2.0),
    ]

    # Initialize encoder for latent feature space evaluation
    encoder_fn = None
    if tf is not None and os.path.isfile("vaegan_celeba.ckpt"):
        print("[*] Initializing pretrained VAE-GAN encoder for latent feature extraction...")
        sess, X, G, Z, Z_mu, Z_log_sigma, is_training, saver = make_network.make_network()
        saver.restore(sess, "vaegan_celeba.ckpt")
        encoder_fn = lambda img_batch: sess.run(Z_mu, feed_dict={X: img_batch, is_training: False})

    print("\n" + "=" * 90)
    print(f"{'Experiment':<26} | {'Alpha':<5} | {'Pairwise SSIM (v)':<18} | {'Pairwise MAE (^)':<17} | {'Latent Dist (^)':<15}")
    print("=" * 90)

    results = []
    for label, exp_dir, alpha in experiments:
        metrics = evaluate_experiment_directory(exp_dir, encoder_fn=encoder_fn)
        results.append((label, alpha, metrics))
        print(f"{label:<26} | {alpha:<5.1f} | {metrics['pairwise_ssim_mean']:.4f} +- {metrics['pairwise_ssim_std']:.4f}   | "
              f"{metrics['pairwise_mae_mean']:.4f} +- {metrics['pairwise_mae_std']:.4f}  | "
              f"{metrics['latent_euc_dist_mean']:.2f}")

    print("=" * 90)
    print("Note: (v) lower indicates greater diversity; (^) higher indicates greater diversity.")

    print("\n--- Identity Preservation Relative to Input Reconstruction ---")
    print(f"{'Experiment':<26} | {'SSIM to Recon':<14} | {'MAE to Recon':<13} | {'SSIM to Input':<14}")
    print("-" * 75)
    for label, alpha, m in results:
        print(f"{label:<26} | {m['recon_ssim_mean']:.4f}         | {m['recon_mae_mean']:.4f}       | {m['input_ssim_mean']:.4f}")
    print("-" * 75)


if __name__ == "__main__":
    main()


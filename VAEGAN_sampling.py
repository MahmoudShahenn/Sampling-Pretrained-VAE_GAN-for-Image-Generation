#!/usr/bin/env python3
"""VAE-GAN Face Sampling and Reconstruction.

Encodes an input facial image into latent space using a pretrained VAE-GAN,
generates 10 diverse but domain-consistent facial variations via controlled
latent sampling, and saves individual sample images, reconstruction, and
visualization grids.

Usage example:
    python VAEGAN_sampling.py --input example.jpg --num_samples 10 --output outputs/
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import make_network

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
except (ImportError, AttributeError):
    try:
        import tensorflow as tf
    except ImportError:
        tf = None


def parse_arguments(argv=None):
    """Parse command-line arguments for VAE-GAN sampling."""
    parser = argparse.ArgumentParser(
        description="Sample controlled facial variations from a single input image using pretrained VAE-GAN."
    )
    parser.add_argument(
        "-i", "--input", "--ifile",
        dest="input",
        type=str,
        required=True,
        help="Path to the input face image (JPG, JPEG, or PNG)."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str,
        default=None,
        help="Directory to save generated images and grids (defaults to outputs/ or {filename}_output)."
    )
    parser.add_argument(
        "-n", "--num_samples",
        dest="num_samples",
        type=int,
        default=10,
        help="Number of facial variations to generate (default: 10)."
    )
    parser.add_argument(
        "-c", "--checkpoint",
        dest="checkpoint",
        type=str,
        default="vaegan_celeba.ckpt",
        help="Path to pretrained model checkpoint (default: vaegan_celeba.ckpt)."
    )
    parser.add_argument(
        "--crop_factor",
        type=float,
        default=0.8,
        help="Face center crop factor between 0.1 and 1.0 (default: 0.8)."
    )
    parser.add_argument(
        "--variation_strength",
        type=float,
        default=1.0,
        help="Perturbation strength scale alpha for latent sampling (default: 1.0)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling."
    )

    return parser.parse_args(argv)


def check_checkpoint_exists(ckpt_path):
    """Check if the checkpoint file or its TensorFlow companion files exist."""
    if os.path.isfile(ckpt_path):
        return True
    # TF v1/v2 checkpoints may have .index or .meta files
    if os.path.isfile(ckpt_path + ".index") or os.path.isfile(ckpt_path + ".meta"):
        return True
    return False


def save_grid_figure(images, titles, save_path, figsize=None):
    """Save a list of images side-by-side into a single grid image."""
    n = len(images)
    if figsize is None:
        figsize = (2 * n, 2.4 if titles else 2.0)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.imshow(images[idx])
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=9)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.85 if titles else 0.98, bottom=0.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_arguments(argv)

    # 1. Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input image '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Determine output directory
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output if args.output else f"Results/{input_stem}"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Validate checkpoint
    if not check_checkpoint_exists(args.checkpoint):
        print(
            f"Error: Pretrained checkpoint '{args.checkpoint}' not found.\n"
            f"The VAE-GAN inference pipeline requires pretrained CelebA weights.\n"
            f"Expected checkpoint file: {args.checkpoint}\n"
            f"Please place '{args.checkpoint}' in the project root or specify --checkpoint <path>.\n"
            f"See README.md for details on acquiring the checkpoint.",
            file=sys.stderr
        )
        sys.exit(1)

    # 4. Check TensorFlow availability
    if tf is None:
        print(
            "Error: TensorFlow is not installed in the active environment.\n"
            "This project requires TensorFlow (e.g. tensorflow==1.15 in Python <=3.7 "
            "or modern tensorflow with compat.v1).\n"
            "Please install dependencies via: pip install -r requirements.txt",
            file=sys.stderr
        )
        sys.exit(1)

    # 5. Load and preprocess input image
    print(f"[*] Loading and preprocessing input image: {args.input}")
    try:
        input_img = utils.preprocess128(args.input, crop_factor=args.crop_factor)
    except Exception as e:
        print(f"Error during image preprocessing: {e}", file=sys.stderr)
        sys.exit(1)

    input_tensor = input_img[np.newaxis, ...]  # Shape: (1, 128, 128, 3)

    # 6. Build network and restore pretrained weights
    print("[*] Building VAE-GAN computational graph...")
    sess, X, G, Z, Z_mu, Z_log_sigma, is_training, saver = make_network.make_network()

    print(f"[*] Restoring checkpoint from '{args.checkpoint}'...")
    try:
        saver.restore(sess, args.checkpoint)
        print("[+] Pretrained model restored successfully.")
    except Exception as e:
        print(f"Error restoring model checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    # 7. Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # 8. Encode input image to latent posterior
    print("[*] Encoding input image into latent space...")
    z_mu, z_log_sigma = sess.run([Z_mu, Z_log_sigma], feed_dict={X: input_tensor, is_training: False})
    z_sigma = np.exp(z_log_sigma)

    # 9. Reconstruct input image (mean z with no perturbation)
    print("[*] Generating direct reconstruction...")
    recon_raw = sess.run(G, feed_dict={Z: z_mu, is_training: False})
    recon_img = utils.deprocess_image(recon_raw)
    recon_path = os.path.join(out_dir, f"{input_stem}_reconstruction.jpg")
    plt.imsave(recon_path, recon_img)
    print(f"[+] Saved reconstruction -> {recon_path}")

    # 10. Generate controlled variations
    num_samples = max(1, args.num_samples)
    alpha = float(args.variation_strength)
    print(f"[*] Generating {num_samples} controlled facial variations (alpha={alpha:.2f})...")
    generated_images = []

    for i in range(num_samples):
        # Independent standard Gaussian noise per sample
        noise = np.random.normal(0.0, 1.0, size=z_mu.shape)

        # Posterior-guided latent sampling with uniform alpha across the run
        z_sample = z_mu + alpha * (z_sigma * noise)

        # Decode latent sample to generated image
        gen_raw = sess.run(G, feed_dict={Z: z_sample, is_training: False})
        gen_img = utils.deprocess_image(gen_raw)
        generated_images.append(gen_img)

        # Save individual sample image
        sample_path = os.path.join(out_dir, f"{input_stem}_sample_{i}.jpg")
        plt.imsave(sample_path, gen_img)
        print(f"    Sample {i+1}/{num_samples} (alpha={alpha:.2f}) -> {sample_path}")

    # 11. Save 10-sample grid (compatible with original format)
    grid_path = os.path.join(out_dir, f"{input_stem}_samples_grid.jpg")
    save_grid_figure(
        images=generated_images,
        titles=[f"Sample {i}" for i in range(num_samples)],
        save_path=grid_path
    )
    print(f"[+] Saved samples grid -> {grid_path}")

    # 12. Save comprehensive comparison grid (Input | Reconstruction | Samples)
    comp_path = os.path.join(out_dir, f"{input_stem}_comparison_grid.jpg")
    all_imgs = [input_img, recon_img] + generated_images
    all_titles = ["Input", "Recon"] + [f"S{i}" for i in range(num_samples)]
    save_grid_figure(
        images=all_imgs,
        titles=all_titles,
        save_path=comp_path
    )
    print(f"[+] Saved comparison grid -> {comp_path}")
    print("[+] Done!")


if __name__ == "__main__":
    main()

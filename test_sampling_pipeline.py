"""Unit and validation tests for the VAE-GAN sampling project."""

import os
import tempfile
import unittest
import numpy as np
from PIL import Image

import utils
import VAEGAN_sampling


class TestPreprocessing(unittest.TestCase):
    """Test image loading, format conversion, and preprocessing."""

    def test_rgb_image(self):
        img = Image.new('RGB', (180, 220), color=(120, 80, 200))
        processed = utils.preprocess128(img, crop_factor=0.8)
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0.0 <= processed.min() <= processed.max() <= 1.0)

    def test_rgba_image(self):
        img = Image.new('RGBA', (200, 200), color=(255, 100, 50, 128))
        processed = utils.preprocess128(img, crop_factor=0.8)
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertTrue(0.0 <= processed.min() <= processed.max() <= 1.0)

    def test_grayscale_image(self):
        img = Image.new('L', (150, 150), color=128)
        processed = utils.preprocess128(img, crop_factor=0.8)
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertTrue(0.0 <= processed.min() <= processed.max() <= 1.0)

    def test_numpy_array_input(self):
        # 2D grayscale array
        arr_2d = np.full((100, 100), 128, dtype=np.uint8)
        res_2d = utils.preprocess128(arr_2d)
        self.assertEqual(res_2d.shape, (128, 128, 3))

        # 3D float array [0, 1]
        arr_3d = np.random.rand(150, 120, 3).astype(np.float32)
        res_3d = utils.preprocess128(arr_3d)
        self.assertEqual(res_3d.shape, (128, 128, 3))

    def test_file_formats_jpg_png(self):
        with tempfile.TemporaryDirectory() as td:
            for ext in ['jpg', 'jpeg', 'png']:
                path = os.path.join(td, f"face.{ext}")
                Image.new('RGB', (160, 200), color=(100, 150, 200)).save(path)
                res = utils.preprocess128(path, crop_factor=0.8)
                self.assertEqual(res.shape, (128, 128, 3))

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            utils.preprocess128("nonexistent_image_file.jpg")


class TestDeprocessing(unittest.TestCase):
    """Test decoder output normalization into [0, 1]."""

    def test_already_normalized(self):
        arr = np.array([[[[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]],
                         [[0.3, 0.6, 0.7], [0.4, 0.5, 0.6]]]], dtype=np.float32)
        out = utils.deprocess_image(arr)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertAlmostEqual(out[0, 0, 0], 0.1, places=5)
        self.assertAlmostEqual(out[0, 0, 1], 0.5, places=5)
        self.assertAlmostEqual(out[0, 0, 2], 0.9, places=5)

    def test_negative_values(self):
        arr = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32).reshape(1, 1, 3)
        out = utils.deprocess_image(arr)
        self.assertAlmostEqual(out.min(), 0.0, places=5)
        self.assertAlmostEqual(out.max(), 1.0, places=5)

    def test_overshoot_values(self):
        arr = np.array([[0.0, 5.0, 10.0]], dtype=np.float32).reshape(1, 1, 3)
        out = utils.deprocess_image(arr)
        self.assertAlmostEqual(out.min(), 0.0, places=5)
        self.assertAlmostEqual(out.max(), 1.0, places=5)


class TestCliAndUtilities(unittest.TestCase):
    """Test CLI argument parsing and helper routines."""

    def test_parse_arguments_default(self):
        args = VAEGAN_sampling.parse_arguments(["-i", "example.jpg"])
        self.assertEqual(args.input, "example.jpg")
        self.assertEqual(args.num_samples, 10)
        self.assertEqual(args.checkpoint, "vaegan_celeba.ckpt")
        self.assertAlmostEqual(args.crop_factor, 0.8)
        self.assertAlmostEqual(args.variation_strength, 1.0)
        self.assertIsNone(args.seed)

    def test_parse_arguments_custom(self):
        args = VAEGAN_sampling.parse_arguments([
            "--input", "test.png",
            "--output", "my_outputs/",
            "--num_samples", "5",
            "--checkpoint", "custom.ckpt",
            "--crop_factor", "0.9",
            "--variation_strength", "0.7",
            "--seed", "123"
        ])
        self.assertEqual(args.input, "test.png")
        self.assertEqual(args.output, "my_outputs/")
        self.assertEqual(args.num_samples, 5)
        self.assertEqual(args.checkpoint, "custom.ckpt")
        self.assertAlmostEqual(args.crop_factor, 0.9)
        self.assertAlmostEqual(args.variation_strength, 0.7)
        self.assertEqual(args.seed, 123)

    def test_save_grid_figure(self):
        with tempfile.TemporaryDirectory() as td:
            imgs = [np.ones((128, 128, 3), dtype=np.float32) for _ in range(10)]
            grid_path = os.path.join(td, "test_grid.jpg")
            VAEGAN_sampling.save_grid_figure(imgs, [f"S{i}" for i in range(10)], grid_path)
            self.assertTrue(os.path.isfile(grid_path))
            self.assertGreater(os.path.getsize(grid_path), 0)


if __name__ == "__main__":
    unittest.main()

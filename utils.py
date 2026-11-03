"""Utility functions for VAE-GAN preprocessing, postprocessing, and network layers.

Adapted from:
Parag K. Mital - Creative Applications of Deep Learning
VanRullen & Reddy (2019) - VAE-GAN-CelebA
"""

import os
import numpy as np
from PIL import Image

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    try:
        import tensorflow as tf
    except ImportError:
        tf = None


# ==============================================================================
# Image Preprocessing and Postprocessing
# ==============================================================================

def preprocess128(img_or_path, crop_factor=0.8):
    """
    Load and preprocess an input facial image to (128, 128, 3) normalized in [0, 1].

    Supports JPG, JPEG, and PNG formats, handling RGB, RGBA, and grayscale.
    Applies center square cropping with crop_factor (default 0.8 to focus on the face)
    and resizes without aspect ratio distortion.

    Parameters
    ----------
    img_or_path : str, PIL.Image.Image, or np.ndarray
        File path to the image, a PIL Image object, or a numpy array.
    crop_factor : float, optional
        Fraction of the center region to crop (default 0.8).

    Returns
    -------
    np.ndarray
        Float32 image array of shape (128, 128, 3) with values in [0.0, 1.0].
    """
    if isinstance(img_or_path, (str, bytes, os.PathLike)):
        if not os.path.isfile(img_or_path):
            raise FileNotFoundError(f"Input image file '{img_or_path}' does not exist.")
        pil_img = Image.open(img_or_path).convert('RGB')
    elif isinstance(img_or_path, Image.Image):
        pil_img = img_or_path.convert('RGB')
    elif isinstance(img_or_path, np.ndarray):
        arr = img_or_path
        if arr.ndim == 2:  # Grayscale (H, W) -> (H, W, 3)
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[-1] == 1:  # (H, W, 1) -> (H, W, 3)
            arr = np.repeat(arr, 3, axis=-1)

        # Scale float [0, 1] arrays to uint8 [0, 255] for PIL processing
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        pil_img = Image.fromarray(arr).convert('RGB')
    else:
        raise TypeError(f"Unsupported image input type: {type(img_or_path)}")

    w, h = pil_img.size
    crop_size = int(min(w, h) * crop_factor)
    if crop_size < 1:
        crop_size = min(w, h)

    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    cropped = pil_img.crop((left, top, left + crop_size, top + crop_size))

    # Resize to model input size (128, 128)
    resample_filter = getattr(Image, 'Resampling', Image).BILINEAR
    resized = cropped.resize((128, 128), resample_filter)

    return np.asarray(resized, dtype=np.float32) / 255.0


def deprocess_image(g):
    """
    Normalize decoder output array into standard [0, 1] RGB image range.

    Parameters
    ----------
    g : np.ndarray
        Decoder output tensor/array of shape (1, H, W, 3) or (H, W, 3).

    Returns
    -------
    np.ndarray
        Normalized image array of shape (H, W, 3) with float32 values in [0.0, 1.0].
    """
    arr = np.asarray(g, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4:
        arr = arr.reshape((-1,) + arr.shape[-2:])

    val_min = float(arr.min())
    val_max = float(arr.max())

    if val_max > val_min:
        if val_min < 0.0 or val_max > 1.0:
            arr = (arr - val_min) / (val_max - val_min)
        else:
            arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)

    return arr


# ==============================================================================
# Neural Network Layer Helpers (TensorFlow 1.x & 2.x compat.v1)
# ==============================================================================

def _get_initializer():
    """Return Xavier / Glorot uniform weight initializer compatible across TF versions."""
    if tf is None:
        raise ImportError("TensorFlow is required for network layer operations.")
    if hasattr(tf, 'glorot_uniform_initializer'):
        return tf.glorot_uniform_initializer()
    if hasattr(tf, 'initializers') and hasattr(tf.initializers, 'glorot_uniform'):
        return tf.initializers.glorot_uniform()
    if hasattr(tf, 'contrib') and hasattr(tf.contrib, 'layers'):
        return tf.contrib.layers.xavier_initializer()
    return tf.zeros_initializer()


def conv2d(x, n_output, k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', reuse=None):
    """Helper for creating a 2D convolution operation."""
    if tf is None:
        raise ImportError("TensorFlow is required for conv2d.")

    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=_get_initializer())

        conv = tf.nn.conv2d(
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding,
            name='conv')

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=conv, bias=b)

    return h, W


def deconv2d(x, n_output_h, n_output_w, n_output_ch, n_input_ch=None,
             k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME',
             name='deconv2d', reuse=None):
    """Helper for creating a 2D transposed convolution (deconvolution) operation."""
    if tf is None:
        raise ImportError("TensorFlow is required for deconv2d.")

    with tf.variable_scope(name or 'deconv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, n_output_ch, n_input_ch or x.get_shape()[-1]],
            initializer=_get_initializer())

        output_shape = tf.stack([tf.shape(x)[0], n_output_h, n_output_w, n_output_ch])
        conv = tf.nn.conv2d_transpose(
            value=x,
            filter=W,
            output_shape=output_shape,
            strides=[1, d_h, d_w, 1],
            padding=padding,
            name='conv_t')

        conv.set_shape([None, n_output_h, n_output_w, n_output_ch])

        b = tf.get_variable(
            name='b',
            shape=[n_output_ch],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=conv, bias=b)

    return h, W


def linear(x, n_output, name=None, activation=None, reuse=None):
    """Helper for creating a fully connected layer."""
    if tf is None:
        raise ImportError("TensorFlow is required for linear layer.")

    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=_get_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=tf.matmul(x, W), bias=b)

        if activation:
            h = activation(h)

        return h, W


def flatten(x, name=None, reuse=None):
    """Flatten tensor to 2 dimensions [batch_size, features]."""
    if tf is None:
        raise ImportError("TensorFlow is required for flatten.")

    with tf.variable_scope(name or 'flatten', reuse=reuse):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(x, shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) in (1, 2):
            flattened = x
        else:
            raise ValueError(f"Expected tensor rank 1, 2, or 4. Found {len(dims)}")
        return flattened


def batch_norm(x, phase_train, name='bn', decay=0.9, reuse=None, affine=True):
    """Batch normalization for convolutional feature maps."""
    if tf is None:
        raise ImportError("TensorFlow is required for batch_norm.")

    with tf.variable_scope(name, reuse=reuse):
        shape = x.get_shape().as_list()
        beta = tf.get_variable(
            name='beta', shape=[shape[-1]],
            initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable(
            name='gamma', shape=[shape[-1]],
            initializer=tf.constant_initializer(1.0), trainable=affine)

        if len(shape) == 4:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        else:
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(
            phase_train,
            mean_var_with_update,
            lambda: (ema_mean, ema_var))

        if hasattr(tf.nn, 'batch_normalization'):
            normed = tf.nn.batch_normalization(
                x, mean, var, offset=beta, scale=gamma if affine else None,
                variance_epsilon=1e-6)
        else:
            normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, beta, gamma, 1e-6, affine)

    return normed

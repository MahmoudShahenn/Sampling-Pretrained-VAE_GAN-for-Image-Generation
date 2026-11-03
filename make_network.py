"""Build the VAE-GAN model architecture for CelebA facial image generation.

Architecture based on:
VanRullen & Reddy (2019) - Reconstructing Faces from fMRI Patterns using
Deep Generative Neural Networks.
"""

import numpy as np
import utils

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
except (ImportError, AttributeError):
    try:
        import tensorflow as tf
    except ImportError:
        tf = None


def encoder(x, is_training, channels, filter_sizes, activation=None, reuse=None):
    """
    Encode input image through convolutional feature extraction layers.

    Parameters
    ----------
    x : tf.Tensor
        Input image tensor [batch_size, 128, 128, 3].
    is_training : tf.Tensor (bool)
        Flag for batch normalization.
    channels : list of int
        Filter counts per convolutional layer.
    filter_sizes : list of int
        Kernel spatial dimensions per layer.
    activation : callable, optional
        Activation function (defaults to tf.nn.elu).
    reuse : bool, optional
        Variable reuse flag.

    Returns
    -------
    h : tf.Tensor
        Final encoded hidden representation.
    hs : list of tf.Tensor
        Intermediate hidden layer representations.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for encoder.")
    if activation is None:
        activation = tf.nn.elu

    h = x
    hs = []
    for layer_i in range(len(channels)):
        with tf.variable_scope(f'layer{layer_i + 1}', reuse=reuse):
            h, W = utils.conv2d(
                h, channels[layer_i],
                k_h=filter_sizes[layer_i],
                k_w=filter_sizes[layer_i],
                d_h=2, d_w=2,
                reuse=reuse
            )
            h = utils.batch_norm(h, is_training)
            h = activation(h)
            hs.append(h)
    return h, hs


def variational_bayes(h, n_code):
    """
    Compute variational posterior parameters (mean and log variance) and sample z.

    Parameters
    ----------
    h : tf.Tensor
        Hidden representation from encoder dense projection.
    n_code : int
        Dimensionality of latent code (1024).

    Returns
    -------
    z : tf.Tensor
        Sampled latent vector using reparameterization trick.
    z_mu : tf.Tensor
        Latent mean vector.
    z_log_sigma : tf.Tensor
        Latent log standard deviation vector.
    loss_z : tf.Tensor
        KL divergence loss.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for variational_bayes.")

    # Latent mean bounded in [-1, 1] via tanh
    z_mu = tf.nn.tanh(utils.linear(h, n_code, name='mu')[0])
    # Latent log sigma bounded in [-0.5, 0.5] via 0.5 * tanh
    z_log_sigma = 0.5 * tf.nn.tanh(utils.linear(h, n_code, name='log_sigma')[0])

    # Sample epsilon from standard normal distribution
    random_normal_fn = getattr(tf.random, 'normal', tf.random_normal)
    epsilon = random_normal_fn(tf.stack([tf.shape(h)[0], n_code]))

    # Reparameterization trick: z = mu + eps * exp(log_sigma)
    z = z_mu + tf.multiply(epsilon, tf.exp(z_log_sigma))

    # Analytical KL divergence loss: -0.5 * sum(1 + 2*log_sigma - mu^2 - exp(2*log_sigma))
    loss_z = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        axis=1
    )

    return z, z_mu, z_log_sigma, loss_z


def decoder(z, is_training, dimensions, channels, filter_sizes,
            activation=None, reuse=None):
    """
    Decode latent tensor through transposed convolutions to generate 128x128 face.

    Parameters
    ----------
    z : tf.Tensor
        Projected latent spatial tensor [batch_size, 4, 4, 1024].
    is_training : tf.Tensor (bool)
        Flag for batch normalization.
    dimensions : list of int
        Output spatial dimensions per layer [8, 16, 32, 64, 128].
    channels : list of int
        Output channel counts per layer [512, 384, 256, 192, 3].
    filter_sizes : list of int
        Kernel spatial dimensions per layer.
    activation : callable, optional
        Activation function (defaults to tf.nn.elu).
    reuse : bool, optional
        Variable reuse flag.

    Returns
    -------
    h : tf.Tensor
        Generated output image tensor [batch_size, 128, 128, 3].
    """
    if tf is None:
        raise ImportError("TensorFlow is required for decoder.")
    if activation is None:
        activation = tf.nn.elu

    h = z
    for layer_i in range(len(dimensions)):
        with tf.variable_scope(f'layer{layer_i + 1}', reuse=reuse):
            h, W = utils.deconv2d(
                x=h,
                n_output_h=dimensions[layer_i],
                n_output_w=dimensions[layer_i],
                n_output_ch=channels[layer_i],
                k_h=filter_sizes[layer_i],
                k_w=filter_sizes[layer_i],
                reuse=reuse
            )
            h = utils.batch_norm(h, is_training)
            h = activation(h)
    return h


def make_network(return_log_sigma=True):
    """
    Construct the full VAE-GAN computational graph, session, and checkpoint saver.

    Parameters
    ----------
    return_log_sigma : bool, optional
        Whether to return Z_log_sigma alongside Z_mu (default True).

    Returns
    -------
    sess : tf.Session
        Active TensorFlow session.
    X : tf.Tensor
        Input image placeholder [batch, 128, 128, 3].
    G : tf.Tensor
        Generated image output tensor.
    Z : tf.Tensor
        Latent vector tensor / generator input.
    Z_mu : tf.Tensor
        Latent mean vector tensor.
    Z_log_sigma : tf.Tensor (if return_log_sigma=True)
        Latent log variance/sigma tensor.
    is_training : tf.Tensor (bool)
        Training phase flag placeholder.
    saver : tf.train.Saver
        Model weights saver / restorer.
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to build the VAE-GAN computational graph. "
            "Please ensure TensorFlow is installed."
        )

    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='istraining')

    n_pixels = 128
    n_channels = 3
    input_shape = [None, n_pixels, n_pixels, n_channels]

    # Input image placeholder
    X = tf.placeholder(name='X', dtype=np.float32, shape=input_shape)

    # Encoder specification
    channels = [192, 256, 384, 512, 768]
    filter_sizes = [3, 3, 3, 3, 3]
    activation = tf.nn.elu
    n_hidden = 1024

    with tf.variable_scope('encoder'):
        H, _ = encoder(X, is_training, channels, filter_sizes, activation)
        Z_hidden = utils.linear(H, n_hidden)[0]

    n_code = 1024
    with tf.variable_scope('encoder/variational'):
        Z_sample, Z_mu, Z_log_sigma, loss_Z = variational_bayes(h=Z_hidden, n_code=n_code)

    # Latent tensor used by generator (can be fed directly via feed_dict)
    Z = Z_sample

    # Decoder specification
    dimensions = [n_pixels // 16, n_pixels // 8, n_pixels // 4, n_pixels // 2, n_pixels]
    channels = [512, 384, 256, 192, n_channels]
    filter_sizes = [3, 3, 3, 3, 3]
    n_latent = n_code * (n_pixels // 32) ** 2

    with tf.variable_scope('generator'):
        Z_decode = utils.linear(
            Z, n_output=n_latent, name='fc', activation=activation)[0]
        Z_decode_tensor = tf.reshape(
            Z_decode, [-1, n_pixels // 32, n_pixels // 32, n_code], name='reshape')
        G = decoder(
            Z_decode_tensor, is_training, dimensions,
            channels, filter_sizes, activation)

    sess = tf.Session()
    init_op = (tf.global_variables_initializer()
               if hasattr(tf, 'global_variables_initializer')
               else tf.initialize_all_variables())
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)

    if return_log_sigma:
        return sess, X, G, Z, Z_mu, Z_log_sigma, is_training, saver
    return sess, X, G, Z, Z_mu, is_training, saver

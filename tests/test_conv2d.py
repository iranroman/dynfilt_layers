import dynfilt_layers
import numpy as np
import tensorflow as tf
import pytest

paddings = ["VALID", "SAME"]


@pytest.mark.parametrize("padding", paddings)
def test_conv2d_same_as_keras_batchsize1(padding):

    # define a kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0)

    # get the kernel values
    kernel_values = kernel_initializer(shape=(2, 2, 1, 3)).numpy()

    # intialize Keras Conv2D layer with the kernel values
    constant_initializer = tf.constant_initializer(kernel_values)
    keras_conv = tf.keras.layers.Conv2D(
        3,
        (2, 2),
        kernel_initializer=constant_initializer,
        use_bias=False,
        padding=padding,
    )

    # initilize dynfilt_layers.Conv2D
    dyn_conv = dynfilt_layers.Conv2D(padding=padding)

    # create dummy data
    X = np.random.randn(1, 256, 256, 1)

    # convolve with Keras
    Y1 = keras_conv(X)

    # convolve with dynfilt
    Y2 = dyn_conv(X, tf.expand_dims(kernel_values, 0))

    assert np.allclose(Y1, Y2)

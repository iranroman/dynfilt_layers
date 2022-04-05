import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):

    """2D convolution layer with custom filter

    This allows for each datapoint in a batch to be convolved
    with a different filter

    Args:
      padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros evenly
        to the left/right or up/down of the input. When `padding="same"` and
        `strides=1`, the output has the same size as the input.

    Input shape:
      X: the tensor with the data on which the convolution operation will be
        applied. 4+D tensor with shape: `batch_shape + (rows, cols, channels)`
      kernel: the tensor with the kernel of filters that the convoled with X.
        5+D tensor with shape `batch_shape + (kernel_height, kernel_width, nchans_in, nchans_out)`

    Output shape:
      4+D tensor with shape: `batch_shape + (new_rows, new_cols, filters)`
      `rows` and `cols` values might have changed due to padding.

    Implementation based on this stackoverflow answer
    https://stackoverflow.com/a/50213363
    """

    def __init__(self, padding="VALID"):
        super(Conv2D, self).__init__()
        self.padding = padding

    def call(self, X, kernel):

        ########################
        # Kernel preprocessing #
        ########################

        # get input dimensions
        batch_size, k_height, k_width, nchans_in, nchans_out = kernel.shape

        # transpose the kernel
        K = tf.transpose(kernel, [1, 2, 0, 3, 4])

        # reshape the kernel
        K = tf.reshape(K, [k_height, k_width, -1, nchans_out])

        #######################
        # input preprocessing #
        #######################

        # get input dimensions
        _, H, W, _ = X.shape

        # transpose the input
        X = tf.transpose(X, [1, 2, 0, 3])  # shape (H, W, batch_size, nchans_in)

        # reshape the input
        X = tf.reshape(X, [1, H, W, -1])

        ###############################
        # do a depth-wise convolution #
        ###############################
        out = tf.nn.depthwise_conv2d(
            X, filter=K, strides=[1, 1, 1, 1], padding=self.padding
        )  # here no requirement about padding being 'VALID', use whatever you want.

        ######################################################
        # reshape to have the number of output channels last #
        ######################################################
        if self.padding == "SAME":
            out = tf.reshape(out, [H, W, -1, nchans_in, nchans_out])
        if self.padding == "VALID":
            out = tf.reshape(
                out,
                [H - k_height + 1, W - k_width + 1, -1, nchans_in, nchans_out],
            )

        #######################################################
        # transpose and sum along the input channel dimension #
        #######################################################
        out = tf.transpose(
            out, [2, 0, 1, 3, 4]
        )  # shape (batch_size, H, W, nchans_in, nchans_out)
        out = tf.reduce_sum(out, axis=3)  # average across the nchans_in dimension

        return out

import math
import tensorflow as tf

from tensorflow.python.ops import random_ops


# Initializer & Regularizer

def variance_scaling_initializer(factor: float = 2., scale_factor: float = .1,
                                 mode: str = "FAN_AVG", uniform: bool = False,
                                 seed: int = 13371337, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        if shape:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.
            fan_out = 1.

        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)

        if mode == 'FAN_IN':
            n = fan_in
        elif mode == 'FAN_OUT':
            n = fan_out
        else:  # mode == 'FAN_AVG':
            n = (fan_in + fan_out) / 2.

        if uniform:
            limit = math.sqrt(3.0 * factor / n)
            _init = random_ops.random_uniform(shape, -limit, limit,
                                              dtype, seed=seed)
        else:
            trunc_stddev = math.sqrt(1.3 * factor / n)
            _init = random_ops.truncated_normal(shape, 0., trunc_stddev,
                                                dtype, seed=seed)
        return _init * scale_factor
    return _initializer


def orthogonal_regularizer(scale: float):
    def ortho_reg(w):
        _, _, _, c = w.get_shape().as_list()
        w = tf.reshape(w, [-1, c])

        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        ortho_loss = tf.nn.l2_loss(reg)
        return scale * ortho_loss
    return ortho_reg


def orthogonal_regularizer_fully(scale: float):
    def ortho_reg_fully(w):
        _, c = w.get_shape().as_list()

        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        ortho_loss = tf.nn.l2_loss(reg)
        return scale * ortho_loss
    return ortho_reg_fully


# Core ops


def conv2d(x,
           channels: int,
           kernel: int = 4, stride: int = 2, pad: int = 0, dilation_rate: int = 1,
           pad_type: str = "zero", use_bias: bool = True, sn: bool = True,
           scope: str = "conv2d_0"):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad *= 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == "zero":
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == "reflect":
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT")

        if sn:
            if scope.__contains__("generator"):
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=w_init,
                                    regularizer=w_reg)
            else:
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=w_init,
                                    regularizer=None)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), dilations=[1, dilation_rate, dilation_rate, 1],
                             strides=[1, stride, stride, 1], padding="VALID")
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(.0))
                x = tf.nn.bias_add(x, bias)
        else:
            if scope.__contains__("generator"):
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=w_init,
                                     kernel_regularizer=w_reg, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias)
            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=w_init,
                                     kernel_regularizer=None, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias)
        return x


def deconv2d(x,
             channels: int,
             kernel: int = 4, stride: int = 2, padding: str = "SAME",
             use_bias: bool = True, sn: bool = True,
             scope: str = "deconv2d_0"):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == "SAME":
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]],
                                initializer=w_init, regularizer=w_reg)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=w_init,
                                           kernel_regularizer=w_reg,
                                           strides=stride, padding=padding, use_bias=use_bias)
        return x


def dense(x, units: int,
          use_bias: bool = True, sn: bool = True,
          scope: str = "dense_0"):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            if scope.__contains__("generator"):
                w = tf.get_variable("kernel", [channels, units], tf.float32,
                                    initializer=w_init, regularizer=w_reg_fully)
            else:
                w = tf.get_variable("kernel", [channels, units], tf.float32,
                                    initializer=w_init, regularizer=None)

            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            if scope.__contains__("generator"):
                x = tf.layers.dense(x, units=units, kernel_initializer=w_init,
                                    kernel_regularizer=w_reg_fully, use_bias=use_bias)
            else:
                x = tf.layers.dense(x, units=units, kernel_initializer=w_init,
                                    kernel_regularizer=None, use_bias=use_bias)
        return x


def spectral_norm(w, iteration: int = 1):
    w_shape = w.shape.as_list()

    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]],
                        initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


w_init = variance_scaling_initializer()
w_reg = orthogonal_regularizer(1e-4)
w_reg_fully = orthogonal_regularizer_fully(1e-4)

"""TensorFlow ops for array / tensor manipulation."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function, absolute_import

import math

import tensorflow as tf


def split_squeeze(dim, num_split, tensor_in):
    """Splits input on given dimension and then squeezes that dimension.

    Args:
        dim: Dimension to split and squeeze on.
        num_split: integer, the number of ways to split.
        tensor_in: Input tensor of shape [N1, N2, .. Ndim, .. Nx].

    Returns:
        List of tensors [N1, N2, .. Ndim-1, Ndim+1, .. Nx].
    """
    return [tf.squeeze(t, squeeze_dims=[dim]) for t in tf.split(dim, num_split, tensor_in)]


def expand_concat(dim, inputs):
    """Expands inputs on given dimension and then concatenates them.

    Args:
        dim: Dimension to expand and concatenate on.
        inputs: List of tensors of the same shape [N1, ... Nx].

    Returns:
        A tensor of shape [N1, .. Ndim, ... Nx]
    """
    return tf.concat(dim, [tf.expand_dims(t, dim) for t in inputs])


def one_hot_matrix(tensor_in, num_classes, on_value=1.0, off_value=0.0):
    """Encodes indices from given tensor as one-hot tensor.

    TODO(ilblackdragon): Ideally implementation should be
    part of TensorFlow with Eigen-native operation.

    Args:
        tensor_in: Input tensor of shape [N1, N2].
        num_classes: Number of classes to expand index into.
        on_value: Tensor or float, value to fill-in given index.
        off_value: Tensor or float, value to fill-in everything else.
    Returns:
        Tensor of shape [N1, N2, num_classes] with 1.0 for each id in original
        tensor.
    """
    tensor_in = tf.convert_to_tensor(tensor_in)
    sparse_values = tf.to_int64(tf.reshape(tensor_in, [-1, 1]))
    size = tf.shape(sparse_values)[0]
    dims = tf.shape(tensor_in)
    indices = tf.to_int64(tf.reshape(tf.range(0, size), [-1, 1]))
    indices_values = tf.concat(1, [indices, sparse_values])
    outshape = tf.to_int64(expand_concat(0, [size, num_classes]))
    one_hot_vector = tf.sparse_to_dense(indices_values, outshape, on_value, off_value)
    ret = tf.reshape(one_hot_vector, tf.concat(0, [dims, [num_classes]]))
    ret.set_shape(tensor_in.get_shape().concatenate(num_classes))
    return ret


def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
    An initializer.
    """
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
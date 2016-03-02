"""TensorFlow ops for deep neural networks."""
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

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as tf_array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from skflow.ops.array_ops import xavier_init

import skflow


def _linear(args, output_size, bias, bias_start=0.0, scope=None, weight_filler=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """
    assert args
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        if weight_filler is None:
            matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        else:
            matrix = vs.get_variable("Matrix", [total_arg_size, output_size],
                                    initializer=xavier_init(total_arg_size, output_size))
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(tf_array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))
        return res + bias_term



def dnn(tensor_in, hidden_units, activation=tf.nn.relu, keep_prob=None, weight_filler=None):
    """Creates fully connected deep neural network subgraph.

    Args:
        tenson_in: tensor or placeholder for input features.
        hidden_units: list of counts of hidden units in each layer.
        activation: activation function between layers. Can be None.
        keep_proba: if not None, will add a dropout layer with given
                    probability.

    Returns:
        A tensor which would be a deep neural network.
    """
    with tf.variable_scope('dnn'):
        for i, n_units in enumerate(hidden_units):
            with tf.variable_scope('layer%d' % i):
                tensor_in = _linear(tensor_in, n_units, True, weight_filler=weight_filler)
                if activation:
                    tensor_in = activation(tensor_in)
                if keep_prob:
                    tensor_in = skflow.ops.dropout(tensor_in, keep_prob)
        return tensor_in


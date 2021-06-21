# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Dropout, LSTM, Dense, Permute, Reshape, BatchNormalization
from tensorflow.keras.utils import Sequence
#from ..mapper import SlidingFeaturesNodeGenerator
# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#__all__ = [
#    "SlidingFeaturesNodeGenerator",
#    "SlidingFeaturesNodeSequence",
#]

#import numpy as np
#from . import Generator
#from tensorflow.keras.utils import Sequence

#from ..core.validation import require_integer_in_range
import numpy as np

# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc


class Generator(abc.ABC):
    """
    A generator supports creating sequences for input into graph machine learning algorithms via the `flow` method.
    """

    @abc.abstractmethod
    def num_batch_dims(self):
        """
        Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).
        For instance, for full batch methods like GCN, the feature has shape ``1 × number of nodes ×
        feature size``, where the 1 is a "dummy" batch dimension and ``number of nodes`` is the real
        batch size (every node in the graph).
        """
        ...

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        """
        Create a Keras Sequence or similar input, appropriate for a graph machine learning model.
        """
        ...

    def default_corrupt_input_index_groups(self):
        """
        Optionally returns the indices of input tensors that can be shuffled for
        :class:`.CorruptedGenerator` to use in :class:`.DeepGraphInfomax`.
        If this isn't overridden, this method returns None, indicating that the generator doesn't
        have a default or "canonical" set of indices that can be corrupted for Deep Graph Infomax.
        """
        return None

def separated(values, *, limit, stringify, sep):
    """
    Print up to ``limit`` values with a separator.
    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
        sep (str): the separator to use between elements (and the "... (NNN more)" continuation)
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation


def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.
    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    return separated(values, limit=limit, stringify=stringify, sep=", ")


def require_dataframe_has_columns(name, df, columns):
    if not set(columns).issubset(df.columns):
        raise ValueError(
            f"{name}: expected {comma_sep(columns)} columns, found: {comma_sep(df.columns)}"
        )


def require_integer_in_range(x, variable_name, min_val=-np.inf, max_val=np.inf):
    """
    A function to verify that a variable is an integer in a specified closed range.
    Args:
        x: the variable to check
        variable_name (str): the name of the variable to print out in error messages
        min_val: the minimum value that `x` can attain
        min_val: the maximum value that `x` can attain
    """

    if not isinstance(x, int):
        raise TypeError(f"{variable_name}: expected int, found {type(x).__name__}")

    if x < min_val or x > max_val:

        if min_val == -np.inf:
            region = f"<= {max_val}"
        elif max_val == np.inf:
            region = f">= {min_val}"
        else:
            region = f"in the range [{min_val}, {max_val}]"

        raise ValueError(f"{variable_name}: expected integer {region}, found {x}")


class SlidingFeaturesNodeGenerator(Generator):
    """
    A data generator for a graph containing sequence data, created by sliding windows across the
    features of each node in a graph.
    .. seealso:: Model using this generator: :class:`.GCN_LSTM`.
    Args:
        G (StellarGraph): a graph instance where the node features are ordered sequence data
        window_size (int): the number of sequence points included in the sliding window.
        batch_size (int, optional): the number of sliding windows to include in each batch.
    """

    def __init__(self, G, window_size, batch_size=1):
        require_integer_in_range(window_size, "window_size", min_val=1)
        require_integer_in_range(batch_size, "batch_size", min_val=1)

        self.graph = G

        node_type = G.unique_node_type(
            "G: expected a graph with a single node type, found a graph with node types: %(found)s"
        )
        self._features = G.node_features(node_type=node_type)
        if len(self._features.shape) == 3:
            self.variates = self._features.shape[2]
        else:
            self.variates = None

        self.window_size = window_size
        self._batch_size = batch_size

    def num_batch_dims(self):
        return 1

    def flow(self, sequence_iloc_slice, target_distance=None):
        """
        Create a sequence object for time series prediction within the given section of the node
        features.
        This handles both univariate data (each node has a single associated feature vector) and
        multivariate data (each node has an associated feature tensor). The features are always
        sliced and indexed along the first feature axis.
        Args:
            sequence_iloc_slice (slice):
                A slice object of the range of features from which to select windows. A slice object
                is the object form of ``:`` within ``[...]``, e.g. ``slice(a, b)`` is equivalent to
                the ``a:b`` in ``v[a:b]``, and ``slice(None, b)`` is equivalent to ``v[:b]``. As
                with that slicing, this parameter is inclusive in the start and exclusive in the
                end.
                For example, suppose the graph has feature vectors of length 10 and ``window_size =
                3``:
                * passing in ``slice(None, None)`` will create 7 windows across all 10 features
                  starting with the features slice ``0:3``, then ``1:4``, and so on.
                * passing in ``slice(4, 7)`` will create just one window, slicing the three elements
                  ``4:7``.
                For training, one might do a train-test split by choosing a boundary and considering
                everything before that as training data, and everything after, e.g. 80% of the
                features::
                    train_end = int(0.8 * sequence_length)
                    train_gen = sliding_generator.flow(slice(None, train_end))
                    test_gen = sliding_generator.flow(slice(train_end, None))
            target_distance (int, optional):
                The distance from the last element of each window to select an element to include as
                a supervised training target. Note: this always stays within the slice defined by
                ``sequence_iloc_slice``.
                Continuing the example above: a call like ``sliding_generator.flow(slice(4, 9),
                target_distance=1)`` will yield two pairs of window and target:
                * a feature window slicing ``4:7`` which includes the features at indices 4, 5, 6,
                  and then a target feature at index 7 (distance 1 from the last element of the
                  feature window)
                * a feature window slicing ``5:8`` and a target feature from index 8.
        Returns:
            A Keras sequence that yields batches of sliced windows of features, and, optionally,
            selected target values.
        """
        return SlidingFeaturesNodeSequence(
            self._features,
            self.window_size,
            self._batch_size,
            sequence_iloc_slice,
            target_distance,
        )


class SlidingFeaturesNodeSequence(Sequence):
    def __init__(
        self, features, window_size, batch_size, sequence_iloc_slice, target_distance
    ):
        if target_distance is not None:
            require_integer_in_range(target_distance, "target_distance", min_val=1)

        if not isinstance(sequence_iloc_slice, slice):
            raise TypeError(
                f"sequence_iloc_slice: expected a slice(...) object, found {type(sequence_iloc_slice).__name__}"
            )

        if sequence_iloc_slice.step not in (None, 1):
            raise TypeError(
                f"sequence_iloc_slice: expected a slice object with a step = 1, found step = {sequence_iloc_slice.step}"
            )

        self._features = features[:, sequence_iloc_slice, ...]
        shape = self._features.shape
        self._num_nodes = shape[0]
        self._num_sequence_samples = shape[1]
        self._num_sequence_variates = shape[2:]

        self._window_size = window_size
        self._target_distance = target_distance
        self._batch_size = batch_size

        query_length = window_size + (0 if target_distance is None else target_distance)
        self._num_windows = self._num_sequence_samples - query_length + 1

        # if there's not enough data to fill one window, there's a problem!
        if self._num_windows <= 0:
            if target_distance is None:
                target_str = ""
            else:
                target_str = f" + target_distance={target_distance}"

            total_sequence_samples = features.shape[1]
            start, stop, step = sequence_iloc_slice.indices(total_sequence_samples)
            # non-trivial steps aren't supported at the moment, so this doesn't need to be included
            # in the message
            assert step == 1

            raise ValueError(
                f"expected at least one sliding window of features, found a total window of size {query_length} (window_size={window_size}{target_str}) which is larger than the {self._num_sequence_samples} selected feature sample(s) (sequence_iloc_slice selected from {start} to {stop} in the sequence axis of length {total_sequence_samples})"
            )

    def __len__(self):
        return int(np.ceil(self._num_windows / self._batch_size))

    def __getitem__(self, batch_num):
        first_start = batch_num * self._batch_size
        last_start = min((batch_num + 1) * self._batch_size, self._num_windows)

        has_targets = self._target_distance is not None

        arrays = []
        targets = [] if has_targets else None
        for start in range(first_start, last_start):
            end = start + self._window_size
            arrays.append(self._features[:, start:end, ...])
            if has_targets:
                target_idx = end + self._target_distance - 1
                targets.append(self._features[:, target_idx, ...])

        this_batch_size = last_start - first_start

        batch_feats = np.stack(arrays)
        assert (
            batch_feats.shape
            == (this_batch_size, self._num_nodes, self._window_size)
            + self._num_sequence_variates
        )

        if has_targets:
            batch_targets = np.stack(targets)
            assert (
                batch_targets.shape
                == (this_batch_size, self._num_nodes) + self._num_sequence_variates
            )
        else:
            batch_targets = None

        return [batch_feats], batch_targets

#from ..core.experimental import experimental
#from ..core.utils import calculate_laplacian

def calculate_laplacian(adj):
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    return adj

class FixedAdjacencyGraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on https://github.com/tkipf/keras-gcn.
    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn
    Notes:
      - The inputs are 3 dimensional tensors: batch size, sequence length, and number of nodes.
      - This class assumes that a simple unweighted or weighted adjacency matrix is passed to it,
        the normalized Laplacian matrix is calculated within the class.
    Args:
        units (int): dimensionality of output feature vectors
        A (N x N): weighted/unweighted adjacency matrix
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
        self,
        units,
        A,
        activation=None,
        use_bias=True,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.adj = calculate_laplacian(A) if A is not None else None

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by Keras model serialization.
        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            # the adjacency matrix argument is required, but
            # (semi-secretly) supports None for loading from a saved
            # model, where the adjacency matrix is a saved weight
            "A": None,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:
        Args:
            input_shapes (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.
        Returns:
            An input shape tuple.
        """
        feature_shape = input_shapes

        return feature_shape[0], feature_shape[1], self.units

    def build(self, input_shapes):
        """
        Builds the layer
        Args:
            input_shapes (list of int): shapes of the layer's inputs (the batches of node features)
        """
        _batch_dim, n_nodes, features = input_shapes

        if self.adj is not None:
            adj_init = initializers.constant(self.adj)
        else:
            adj_init = initializers.zeros()

        self.A = self.add_weight(
            name="A", shape=(n_nodes, n_nodes), trainable=False, initializer=adj_init
        )
        self.kernel = self.add_weight(
            shape=(features, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                # ensure the per-node bias can be broadcast across each feature
                shape=(n_nodes, 1),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, features):
        """
        Applies the layer.
        Args:
            features (ndarray): node features (size B x N x F), where B is the batch size, F = TV is
                the feature size (consisting of the sequence length and the number of variates), and
                N is the number of nodes in the graph.
        Returns:
            Keras Tensor that represents the output of the layer.
        """

        # Calculate the layer operation of GCN
        # shape = B x F x N
        nodes_last = tf.transpose(features, [0, 2, 1])
        neighbours = K.dot(nodes_last, self.A)

        # shape = B x N x F
        h_graph = tf.transpose(neighbours, [0, 2, 1])
        # shape = B x N x units
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias

        output = self.activation(output)

        return output


#@experimental(
#    reason="Lack of unit tests and code refinement", issues=[1132, 1526, 1564]
#)
class GCN_LSTM:

    """
    GCN_LSTM is a univariate timeseries forecasting method. The architecture  comprises of a stack of N1 Graph Convolutional layers followed by N2 LSTM layers, a Dropout layer, and  a Dense layer.
    This main components of GNN architecture is inspired by: T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction (https://arxiv.org/abs/1811.05320).
    The implementation of the above paper is based on one graph convolution layer stacked with a GRU layer.
    The StellarGraph implementation is built as a stack of the following set of layers:
    1. User specified no. of Graph Convolutional layers
    2. User specified no. of LSTM layers
    3. 1 Dense layer
    4. 1 Dropout layer.
    The last two layers consistently showed better performance and regularization experimentally.
    .. seealso::
       Example using GCN_LSTM: `spatio-temporal time-series prediction <https://stellargraph.readthedocs.io/en/stable/demos/time-series/gcn-lstm-time-series.html>`__.
       Appropriate data generator: :class:`.SlidingFeaturesNodeGenerator`.
       Related model: :class:`.GCN` for graphs without time-series node features.
    Args:
       seq_len: No. of LSTM cells
       adj: unweighted/weighted adjacency matrix of [no.of nodes by no. of nodes dimension
       gc_layer_sizes (list of int): Output sizes of Graph Convolution  layers in the stack.
       lstm_layer_sizes (list of int): Output sizes of LSTM layers in the stack.
       generator (SlidingFeaturesNodeGenerator): A generator instance.
       bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
       dropout (float): Dropout rate applied to input features of each GCN layer.
       gc_activations (list of str or func): Activations applied to each layer's output; defaults to ``['relu', ..., 'relu']``.
       lstm_activations (list of str or func): Activations applied to each layer's output; defaults to ``['tanh', ..., 'tanh']``.
       kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
       kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
       kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
       bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
       bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
       bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
     """

    def __init__(
        self,
        seq_len,
        adj,
        gc_layer_sizes,
        lstm_layer_sizes,
        forecast_horizon,
        gc_activations=None,
        generator=None,
        lstm_activations=None,
        bias=True,
        dropout=0.5,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if generator is not None:
            if not isinstance(generator, SlidingFeaturesNodeGenerator):
                raise ValueError(
                    f"generator: expected a SlidingFeaturesNodeGenerator, found {type(generator).__name__}"
                )

            if seq_len is not None or adj is not None:
                raise ValueError(
                    "expected only one of generator and (seq_len, adj) to be specified, found multiple"
                )

            adj = generator.graph.to_adjacency_matrix(weighted=True).todense()
            seq_len = generator.window_size
            variates = generator.variates
        else:
            variates = None

        super(GCN_LSTM, self).__init__()

        n_gc_layers = len(gc_layer_sizes)
        n_lstm_layers = len(lstm_layer_sizes)

        self.lstm_layer_sizes = lstm_layer_sizes
        self.gc_layer_sizes = gc_layer_sizes
        self.bias = bias
        self.dropout = dropout
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.forecast_horizon = forecast_horizon
        self.n_features = seq_len
        self.seq_len = seq_len
        self.multivariate_input = variates is not None
        self.variates = variates if self.multivariate_input else 1
        self.outputs = self.n_nodes * self.variates

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        # Activation function for each gcn layer
        if gc_activations is None:
            gc_activations = ["relu"] * n_gc_layers
        elif len(gc_activations) != n_gc_layers:
            raise ValueError(
                "Invalid number of activations; require one function per graph convolution layer"
            )
        self.gc_activations = gc_activations

        # Activation function for each lstm layer
        if lstm_activations is None:
            lstm_activations = ["tanh"] * n_lstm_layers
        elif len(lstm_activations) != n_lstm_layers:
            padding_size = n_lstm_layers - len(lstm_activations)
            if padding_size > 0:
                lstm_activations = lstm_activations + ["tanh"] * padding_size
            else:
                raise ValueError(
                    "Invalid number of activations; require one function per lstm layer"
                )
        self.lstm_activations = lstm_activations

        self._gc_layers = [
            FixedAdjacencyGraphConvolution(
                units=self.variates * layer_size,
                A=self.adj,
                activation=activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            )
            for layer_size, activation in zip(self.gc_layer_sizes, self.gc_activations)
        ]
        self._lstm_layers = [
            LSTM(layer_size, activation=activation, return_sequences=True)
            for layer_size, activation in zip(
                self.lstm_layer_sizes[:-1], self.lstm_activations
            )
        ]
        self._lstm_layers.append(
            LSTM(
                self.lstm_layer_sizes[-1],
                activation=self.lstm_activations[-1],
                return_sequences=False#,
                #dropout=0.1
            )
        )
        self._bn = BatchNormalization(),
        self._decoder_layer = Dense(self.n_nodes *self.forecast_horizon)

    def __call__(self, x):

        x_in, out_indices = x

        h_layer = x_in
        if not self.multivariate_input:
            # normalize to always have a final variate dimension, with V = 1 if it doesn't exist
            # shape = B x N x T x 1
            h_layer = tf.expand_dims(h_layer, axis=-1)

        # flatten variates into sequences, for convolution
        # shape B x N x (TV)
        h_layer = Reshape((self.n_nodes, self.seq_len * self.variates))(h_layer)

        for layer in self._gc_layers:
            h_layer = layer(h_layer)

        # return the layer to its natural multivariate tensor form
        # shape B x N x T' x V (where T' is the sequence length of the last GC)
        h_layer = Reshape((self.n_nodes, -1, self.variates))(h_layer)
        # put time dimension first for LSTM layers
        # shape B x T' x N x V
        h_layer = Permute((2, 1, 3))(h_layer)
        # flatten the variates across all nodes, shape B x T' x (N V)
        h_layer = Reshape((-1, self.n_nodes * self.variates))(h_layer)

        for layer in self._lstm_layers:
            h_layer = layer(h_layer)

        #h_layer = Dropout(self.dropout)(h_layer)
        h_layer = BatchNormalization()(h_layer)
        h_layer = self._decoder_layer(h_layer)
        h_layer = Reshape((self.n_nodes ,self.forecast_horizon))(h_layer)

        if self.multivariate_input:
            # flatten things out to the multivariate shape
            # shape B x N x V
            h_layer = Reshape((self.n_nodes, self.variates))(h_layer)

        return h_layer

    def in_out_tensors(self):
        """
        Builds a GCN model for node  feature prediction
        Returns:
            tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras/TensorFlow
                input tensors for the GCN model and ``x_out`` is a tensor of the GCN model output.
        """
        # Inputs for features
        if self.multivariate_input:
            shape = (None, self.n_nodes, self.n_features, self.variates)
        else:
            shape = (None, self.n_nodes, self.n_features)

        x_t = Input(batch_shape=shape)

        # Indices to gather for model output
        out_indices_t = Input(batch_shape=(None, self.n_nodes), dtype="int32")

        x_inp = [x_t, out_indices_t]
        x_out = self(x_inp)

        return x_inp[0], x_out

from tensorflow.keras import Sequential, Model

def get_gcnlstm(forecast_horizon, num_lags, sensor_adj):
    gcn_lstm = GCN_LSTM(
        seq_len=num_lags,
        adj=sensor_adj,
        forecast_horizon=forecast_horizon,
        gc_layer_sizes=[64,64],
        gc_activations=["relu",'relu'],
        lstm_layer_sizes=[120],
        lstm_activations=["tanh"]
    )
    
    x_input, x_output = gcn_lstm.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)
    return model
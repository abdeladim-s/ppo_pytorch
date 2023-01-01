#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions

Longer description of this module.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import scipy.signal as signal
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discounted_cumulative_sum(array, discount):
    """
    https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation/47971187#47971187
        C[i] = R[i] + discount * C[i+1]
        signal.lfilter(b, a, x, axis=-1, zi=None)
        a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                              - a[1]*y[n-1] - ... - a[N]*y[n-N]

    Could be considered as well -> https://github.com/toshas/torch-discounted-cumsum
            """
    r = array[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]


def standardize(array):
    array -= np.mean(array)
    array /= np.std(array)
    return array


def build_neural_net_general(state_size, agent_state_layer, action_size, agent_action_layer, agent_hidden_layers):
    state_layer = agent_state_layer[0]
    state_activation_fct = agent_state_layer[1]
    state_kwargs = agent_state_layer[2]

    if len(agent_action_layer) != 0:
        action_layer = agent_action_layer[0]
        action_activation_fct = agent_action_layer[1]
        action_kwargs = agent_action_layer[2]

        layers = [(state_layer, state_size, state_activation_fct, state_kwargs)] + agent_hidden_layers + \
                 [(action_layer, action_size, action_activation_fct, action_kwargs)]
    else:
        layers = [(state_layer, state_size, state_activation_fct, state_kwargs)] + agent_hidden_layers

    torch_layers = []
    for i in range(len(layers)):
        layer_type = layers[i][0]
        in_features = int(layers[i][1])
        out_features = int(layers[i+1][1]) if i < len(layers) - 1 else action_size
        act_fct = layers[i][2]
        kwargs = layers[i][3]

        torch_layers += [layer_type(in_features, out_features, **kwargs), act_fct()]

    return nn.Sequential(*torch_layers).to(device)


def build_neural_net(sizes, activation, output_activation=nn.Identity):
    """
        inspired from https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/ppo/core.py
        :param sizes:
        :param activation:
        :param output_activation:
        :return:
        """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers).to(device)


# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
# https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict = {
    # np.bool: torch.bool,
    np.uint8.__name__: torch.uint8,
    np.int8.__name__: torch.int8,
    np.int16.__name__: torch.int16,
    np.int32.__name__: torch.int32,
    np.int64.__name__: torch.int64,
    np.float16.__name__: torch.float16,
    np.float32.__name__: torch.float32,
    np.float64.__name__: torch.float64,
    np.complex64.__name__: torch.complex64,
    np.complex128.__name__: torch.complex128
}

if __name__ == '__main__':
    pass

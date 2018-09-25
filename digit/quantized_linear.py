import torch
import math
import numpy as np

from torch.nn.functional import linear



def int_to_hex(x, bit_width):
    return hex(int(np.binary_repr(x, bit_width), 2))


def quantized_linear(input, weight, weight_quantization_scheme, bias=None, bias_quantization_scheme=None):
    weight_quantization_scheme.fp_transform_(weight)
    q_weight = weight_quantization_scheme.q_forward(weight)
    if bias is not None:
        bias_quantization_scheme.fp_transform_(bias)
        q_bias = bias_quantization_scheme.q_forward(bias)
        output = linear(input, q_weight, q_bias)
    else:
        output = linear(input, q_weight)
    return output

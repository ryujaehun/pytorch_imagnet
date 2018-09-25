import math

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from digit.quantization_scheme import WeightQuantizationScheme, ActivationQuantizationScheme, BiasQuantizationScheme
from digit.quantized_linear import quantized_linear

class QuantizedLinear(Module):

    def __init__(self, in_features, out_features, weight_bit_width=32, weight_q_type='FP',
                 bias_bit_width=32, bias_q_type='FP', bias=True):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quantization_scheme = WeightQuantizationScheme(weight_bit_width, weight_q_type)
        self.bias_quantization_scheme = BiasQuantizationScheme(bias_bit_width, bias_q_type)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return quantized_linear(input, self.weight, self.weight_quantization_scheme, self.bias, self.bias_quantization_scheme)



    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

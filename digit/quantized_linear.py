import torch
import math
import numpy as np

from torch.nn.functional import linear

def quantized_linear_hls_bias_string(bias, quantization_scheme, hls_var_name, pe=None):
    int_bias = quantization_scheme.to_int(bias).numpy().astype(int)
    if pe is not None:
        assert(int_bias.shape[0] == 1)
        int_bias = np.vstack([int_bias[0][i::pe] for i in xrange(0, pe)]) #create pe subarrays selecting every pe element
    return hls_matrix_string(int_bias, quantization_scheme, hls_var_name)

def quantized_linear_hls_weight_string(weight, quantization_scheme, hls_var_name, simd=None, pe=None):
    int_weight = quantization_scheme.to_int(weight).numpy().astype(int)
    if pe is not None and simd is not None:
        int_weight_simd_pe = np.zeros(shape=(simd, int_weight.shape[0]/simd, pe, int_weight.shape[1]/pe))
        for b in xrange(0, int_weight.shape[0]/simd):
            for a in xrange(0, simd):
                int_weight_simd_pe[a,b,:,:] = np.vstack([int_weight[b*simd+a][j::pe] for j in xrange(0, pe)])
        return hls_matrix_string(int_weight_simd_pe.astype(int), quantization_scheme, hls_var_name)
    else:
        return hls_matrix_string(int_weight, quantization_scheme, hls_var_name)

def int_to_hex(x, bit_width):
    return hex(int(np.binary_repr(x, bit_width), 2))

def hls_matrix_string(int_x, quantization_scheme, hls_var_name):
    x_bit_width = quantization_scheme.q_params.bit_width
    decl = "const ap_uint<{}> {} ".format(x_bit_width, hls_var_name)
    decl += ''.join(["[{}]".format(int_x.shape[i]) for i in range(int_x.ndim)])
    string_list = [" = "]
    matrix_to_string_list(int_x, int_x.ndim - 1, x_bit_width, string_list)
    decl += ''.join(string_list)
    decl +=";\n"
    return quantization_scheme.q_params.gen_hls_typedef(hls_var_name), decl

def matrix_to_string_list(int_x, dims, bit_width, string_list):
    string_list.append("{ ")
    for i in xrange(0, int_x.shape[0]):
        if dims > 0:
            matrix_to_string_list(int_x[i], dims - 1, bit_width, string_list)
        else:
            string_list.append(str(int_to_hex(int_x[i], bit_width)))
        if i < int_x.shape[0] - 1:
            string_list.append(", \n")
        else:
            string_list.append("\n")
    string_list.append("}\n")
    return

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

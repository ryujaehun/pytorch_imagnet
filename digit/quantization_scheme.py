# Copy

from abc import ABCMeta, abstractmethod, abstractproperty

from functools import partial
import math
from .quantization_impl import *


class QuantizationParams(object):

    def __init__(self, bit_width, q_type):
        self.bit_width = bit_width
        self.q_type = q_type

    def gen_hls_typedef(self, name):
        return "typedef ap_fixed<{} , {}, AP_RND_ZERO, AP_WRAP> t_fixed_{};".format(self.bit_width, self.int_bit_width, name)


class SignedFixedUnitQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, int_bit_width, q_type):
        super(SignedFixedUnitQuantizationParams, self).__init__(bit_width, q_type)
        self.int_bit_width = int_bit_width #including implicit sign bit
        self.frac_bit_width = bit_width - int_bit_width
        self.prescale = 2 ** self.frac_bit_width
        self.postscale = 2 ** (- self.frac_bit_width)
        self.min_val = - (2 ** (bit_width - self.frac_bit_width - 1))
        self.max_val = - self.min_val - self.postscale


class UnsignedFixedUnitQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, q_type):
        super(UnsignedFixedUnitQuantizationParams, self).__init__(bit_width, q_type)
        self.prescale = 2 ** self.bit_width
        self.postscale = 1.0 / self.prescale
        self.min_val = 0.0
        self.max_val = 1.0 - self.postscale


class QuantizationScheme(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, bit_width, q_type, scale_factor=None):
        if bit_width == 32 and q_type == 'FP':
            self.q_params = QuantizationParams(bit_width, q_type=q_type)
        elif q_type == 'UNSIGNED_FIXED_UNIT':
            self.q_params = UnsignedFixedUnitQuantizationParams(bit_width, q_type=q_type)
        elif q_type == 'SIGNED_FIXED_UNIT':
            self.q_params = SignedFixedUnitQuantizationParams(bit_width, int_bit_width=1, q_type=q_type)
        else:
            raise Exception('Unknown activation quantization scheme: {} bit width, {} quantization.'.format(bit_width, q_type))

    @property
    def q_forward(self):
        return partial(self.q_impl.apply, self.q_params)

    @property
    def to_int(self):
        return partial(self.q_impl.to_int, self.q_params)

    @abstractproperty
    def q_impl(self):
        pass


class ActivationQuantizationScheme(QuantizationScheme):

    def __init__(self, bit_width, q_type, scale_factor=None):
        super(ActivationQuantizationScheme, self).__init__(bit_width, q_type, scale_factor)

    @property
    def q_impl(self):
        if self.q_params.bit_width == 32 and self.q_params.q_type == 'FP':
            return Identity
        elif self.q_params.q_type == 'SIGNED_FIXED_UNIT' or self.q_params.q_type == 'UNSIGNED_FIXED_UNIT':
            return FixedUnitActivation
        else:
            raise Exception('Unknown activation quantization scheme: {} bit width, {} quantization.'.format(self.q_params.bit_width, self.q_params.q_type))


class WeightQuantizationScheme(QuantizationScheme):

    def __init__(self, bit_width, q_type, scale_factor=None):
        super(WeightQuantizationScheme, self).__init__(bit_width, q_type, scale_factor)

    @property
    def q_impl(self):
        if self.q_params.bit_width == 32 and self.q_params.q_type == 'FP':
            return Identity
        elif self.q_params.q_type == 'SIGNED_FIXED_UNIT' or self.q_params.q_type == 'UNSIGNED_FIXED_UNIT':
            return FixedUnitWeight
        else:
            raise Exception('Unknown activation quantization scheme: {} bit width, {} quantization.'.format(self.q_params.bit_width, self.q_params.q_type))

    '''
    (Non Autograd) function to apply to full precision
    values during forward
    Performed in place on a Variable's data
    '''
    @property
    def fp_transform_(self):
        return lambda x: None

class BiasQuantizationScheme(WeightQuantizationScheme):

    def __init__(self, bit_width, q_type, scale_factor=None):
        super(BiasQuantizationScheme, self).__init__(bit_width, q_type, scale_factor)

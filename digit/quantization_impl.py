import torch
from torch.autograd.function import Function
from torch.autograd import Variable

class Identity(Function):

    @staticmethod
    def to_int(q_params, x):
        raise Exception

    @staticmethod
    def forward(ctx, q_params, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return None, grad


class FixedUnitWeight(Function):

    @staticmethod
    def to_int(q_params, x):
        return x.clamp(q_params.min_val, q_params.max_val).mul(q_params.prescale).round()

    @staticmethod
    def forward(ctx, q_params, x):
        return FixedUnitWeight.to_int(q_params, x).mul(q_params.postscale)

    @staticmethod
    def backward(ctx, grad):
        return None, grad


class FixedUnitActivation(FixedUnitWeight):

    @staticmethod
    def forward(ctx, q_params, x):
        ctx.save_for_backward(x)
        ctx.q_params = q_params
        return super(FixedUnitActivation, FixedUnitActivation).forward(ctx, q_params, x)

    @staticmethod
    def backward(ctx, grad):
        q_params = ctx.q_params
        x, = ctx.saved_tensors

        min_tensor = x.new([q_params.min_val])
        max_tensor = x.new([q_params.max_val])

        #Mask has to be a Variable with a float tensor data
        mask = x.le(max_tensor) * x.ge(min_tensor)
        mask = Variable(mask.type(type(x)), requires_grad=False)

        return None, grad * mask

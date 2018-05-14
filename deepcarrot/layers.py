import numpy as np
from . import core
from .core import Variable


def initialize_he(in_dims, out_dims):
    return np.random.uniform(size=(out_dims, in_dims), low=-1./np.sqrt(out_dims), high=1./np.sqrt(out_dims))


class Linear(object):
    def __init__(self, in_dims, out_dims):
        self.weight = Variable(initialize_he(in_dims, out_dims))
        self.bias = Variable(np.random.uniform(size=out_dims)*0.01)

    def __call__(self, inputs):
        out = core.matmatmul(inputs, self.weight.t()) + self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias]


class Relu(object):
    def __call__(self, inputs):
        out = inputs.relu()
        return out

    def parameters(self):
        return []

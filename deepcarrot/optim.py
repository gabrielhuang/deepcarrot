import numpy as np


class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, grads):
        raise Exception('Not implemented')


class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0.):
        self.lr = lr
        self.momentum = momentum
        self.avg_grads = {}
        self.iterations = 0
        Optimizer.__init__(self, parameters)

    def step(self, grads):
        self.iterations += 1  # for unbiased momentum

        for p in self.parameters:
            grad = grads[p]
            assert grad.shape == p.data.shape

            if p not in self.avg_grads:
                self.avg_grads[p] = np.zeros_like(grad)

            self.avg_grads[p] = self.momentum*self.avg_grads[p] + (1-self.momentum)*grads[p]
            # Unbias
            unbiased_avg_grad = self.avg_grads[p] / (1. - self.momentum**self.iterations)
            p.data -= unbiased_avg_grad * self.lr

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


class Adam(Optimizer):
    def __init__(self, parameters, lr, momentum=0.9, momentum_sqr=0.999, epsilon=1e-8):
        self.lr = lr
        self.momentum = momentum
        self.momentum_sqr = momentum_sqr
        self.epsilon = epsilon

        # Running averages
        self.avg_grads = {}
        self.avg_sqr_grads = {}
        self.iterations = 0
        Optimizer.__init__(self, parameters)

    def step(self, grads):
        self.iterations += 1  # for unbiased momentum

        for p in self.parameters:
            grad = grads[p]
            assert grad.shape == p.data.shape

            if p not in self.avg_grads:
                self.avg_grads[p] = np.zeros_like(grad)
                self.avg_sqr_grads[p] = np.zeros_like(grad)

            self.avg_grads[p] = self.momentum*self.avg_grads[p] + (1-self.momentum)*grads[p]
            self.avg_sqr_grads[p] = self.momentum_sqr*self.avg_sqr_grads[p] + (1-self.momentum_sqr)*grads[p]**2
            # Unbias
            unbiased_avg_grad = self.avg_grads[p] / (1. - self.momentum**self.iterations)
            unbiased_avg_sqr_grad = self.avg_sqr_grads[p] / (1. - self.momentum_sqr**self.iterations)
            # Update
            update =  unbiased_avg_grad / np.sqrt(unbiased_avg_sqr_grad + self.epsilon)
            p.data -= self.lr * update


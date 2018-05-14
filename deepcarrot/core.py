import numpy as np
from collections import OrderedDict
import copy


DEFAULT_VAR = '<var>'


class AbstractVariable(object):

    def __init__(self, data, name=DEFAULT_VAR):
        self.data = np.asarray(data)
        self.name = name

    def __repr__(self):
        if self.name:
            return 'Variable[{}]: {}'.format(self.name, self.data)
        else:
            return 'Variable: {}'.format(self.data)

    def _get_grad(self, contributions, grads_cache):
        total_grad = 0.
        assert self in contributions, 'Variable {} not used to compute loss'.format(self)
        for output, jacobian in contributions[self]:
            if output in grads_cache:
                grad = grads_cache[output]
            else:
                grad = output._get_grad(contributions, grads_cache)

            total_grad += jacobian(grad)
            # here it could be replaced with apply
        grads_cache[self] = total_grad
        return total_grad

    def size(self):
        return self.data.shape

    # Syntactic sugar
    def t(self):
        '''
        Transpose a matrix. Only works for 2D matrices.

        :return:
        '''
        return ComputedVariable([self],
                                TransposeOperator(),
                                name='Transpose({})'.format(self.name))

    def __mul__(self, other):
        return ComputedVariable([self, other],
                                MultiplyOperator(),
                                name='Mul({},{})'.format(self.name, other.name))

    def __add__(self, other):
        return ComputedVariable([self, other],
                                AddOperator(),
                                name='Add({},{})'.format(self.name, other.name))

    def __rshift__(self, f):
        return f(self)


    def relu(self):
        return ComputedVariable([self],
                                ReluOperator(),
                                name='Relu({})'.format(self.name))

    def sum(self, axis=None):
        return ComputedVariable([self],
                                SumOperator(),
                                name='Sum({})'.format(self.name))


class LeafVariable(AbstractVariable):

    def __init__(self, data, name=DEFAULT_VAR):
        AbstractVariable.__init__(self, data, name)

    def _get_contributions(self, contributors=None):
        raise Exception('Cannot differentiate leaf variable.')


Variable = LeafVariable  # shorthand


class ComputedVariable(AbstractVariable):

    def __init__(self, inputs, operator, name=DEFAULT_VAR):
        self.inputs = inputs
        self.operator = operator
        value = self.operator([input.data for input in self.inputs])
        AbstractVariable.__init__(self, value, name)

    def _get_contributions(self, contributors=None):
        '''
        This is only called once per ancestor of the loss node
        '''
        # Get Jacobians
        jacobians = self.operator.get_jacobians([input.data for input in self.inputs])

        contributors = contributors or {}
        # contributors is mutable
        for input, jacobian in zip(self.inputs, jacobians):
            input_registered = (input in contributors)
            contributors.setdefault(input, [])
            contributors[input].append((self, jacobian))
            if not input_registered and isinstance(input, ComputedVariable):
                input._get_contributions(contributors)
        return contributors


class Operator(object):

    def __call__(self, inputs):
        inputs = [np.asarray(input, dtype=np.float32) for input in inputs]
        return self._call(inputs)

    def get_jacobians(self, inputs):
        inputs = [np.asarray(input, dtype=np.float32) for input in inputs]
        return self._get_jacobians(inputs)

    def _call(self, inputs):
        raise Exception('Implement forward pass.')

    def _get_jacobians(self, inputs):
        raise Exception('Implement backward pass.')


class Jacobian(object):

    def __init__(self, function, in_shape, out_shape):
        '''
        Build an abstract representation of a Jacobian as a linear function.

        :param function: Either linear function or 2D np.ndarray
        :param jacobian_shape:
        '''
        assert not isinstance(function, np.ndarray), 'Matrix function not supported anymore. Please provide function'
        #if not isinstance(function, np.ndarray):  # is function
        #    if jacobian_shape is None:
        #        raise ArgumentError('Provide jacobian shape when "function" is not a numpy array.')
        #else:  # is array
        #    jacobian_shape = function.shape
        #assert len(jacobian_shape) == 2, 'Jacobian must represent 2D matrix acting on flattened vectors.'
        self.function = function
        self.in_shape = in_shape
        self.out_shape = out_shape

    def jacobian_shape(self):
        return (prod(self.out_shape), prod(self.in_shape))

    def to_matrix(self):
        '''
        This is for testing. Don't use it for computation, the current implementation is slooooow.

        It represents the Jacobian explicitly as a 2D np.ndarray of shape (output_size, input_size).
        It
        :return: A jacobian matrix of shape jacobian_shape
        '''
        if isinstance(self.function, np.ndarray):
            return self.function
        else:
            possible_grads = np.identity(self.jacobian_shape()[0])
            inputs = []
            for grad in possible_grads:
                input = self.function(grad.reshape(self.out_shape)).flatten()
                inputs.append(input)
            return np.asarray(inputs)

    def backward(self, grad):
        '''
        Left-Multiply a gradient by the transpose of
        :param grad:
        :return:
        '''
        if isinstance(self.function, np.ndarray):
            return self.function.T.dot(grad)  # need transpose because grad.shape == [jacobian_shape.size()[0]]
        else:
            return self.function(grad)

    def __call__(self, grad):
        return self.backward(grad)


class MultiplyOperator(Operator):

    def _call(self, (a, b)):
        return a*b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(lambda x: x*b, a.shape, a.shape)
        jacobian_b = Jacobian(lambda x: x*a, b.shape, b.shape)
        return [jacobian_a, jacobian_b]


class AddOperator(Operator):  # create operator later

    def _call(self, (a, b)):
        return a+b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(lambda grad: grad, a.shape, a.shape)
        jacobian_b = Jacobian(lambda grad: grad, b.shape, b.shape)
        return [jacobian_a, jacobian_b]


class TransposeOperator(Operator):

    def _call(self, (a,)):
        assert len(a.shape) == 2
        return a.T

    def _get_jacobians(self, (a,)):
        jacobian_a = Jacobian(lambda grad: grad.T, a.shape, a.T.shape)
        return [jacobian_a]


# Handles empty tuples correctly
def prod(shape):
    if len(shape) == 0:
        return 1
    return int(np.prod(shape))


class MatrixMatrixOperator(Operator):

    def _call(self, (a, b)):
        assert len(a.shape) == 2 and len(b.shape) == 2
        return a.dot(b)

    def _get_jacobians(self, (a, b)):
        out_shape = (a.shape[0], b.shape[1])
        jacobian_a = Jacobian(lambda grad: grad.dot(b.T), a.shape, out_shape)
        #def closure(grad):
        #    return a.T.dot(grad)
        #jacobian_b = Jacobian(closure, b.shape, out_shape)
        jacobian_b = Jacobian(lambda grad: a.T.dot(grad), b.shape, out_shape)
        return [jacobian_a, jacobian_b]


class ReluOperator(Operator):

    def _call(self, (a,)):
        return np.maximum(a, 0)

    def _get_jacobians(self, (a,)):
        jacobian_a = Jacobian(lambda grad: grad * (a > 0).astype(np.float32), a.shape, a.shape)
        return [jacobian_a]


class SumOperator(Operator):

    def __init__(self, axis=None):
        '''
        Sum and reduce, either over one or all axes.
        :param axis:
        '''
        self.axis = axis

    def _call(self, (a,)):
        return a.sum(axis=self.axis)

    def _get_jacobians(self, (a,)):
        if self.axis is None:
            # This assumes grad is scalar (shape is empty tuple () )
            return [Jacobian(lambda grad: grad * np.ones_like(a), a.shape, ())]
        else:
            out_shape = list(a.shape)
            del out_shape[self.axis]

            def closure(grad):
                grad = np.expand_dims(grad, self.axis)
                return np.repeat(grad, a.shape[self.axis], self.axis)

            jac = Jacobian(closure, a.shape, out_shape)
            return [jac]


class LogSoftmaxOperator(Operator):

    def __init__(self, axis):
        '''
        Softmax reduce over an axis.
        '''
        self.axis = axis

    def _call(self, (a,)):
        # Get maximum and subtract it
        maxes = np.max(a, self.axis)
        #b =
        return a.sum(axis=self.axis)

    def _get_jacobians(self, (a,)):
        if self.axis is None:
            # This assumes grad is scalar (shape is empty tuple () )
            return [Jacobian(lambda grad: grad * np.ones_like(a), a.shape, ())]
        else:
            out_shape = list(a.shape)
            del out_shape[self.axis]
            grad = np.exp

            def closure(grad):
                grad = np.expand_dims(grad, self.axis)
                return np.repeat(grad, a.shape[self.axis], self.axis),
            jac = Jacobian(lambda grad: np.repeat(grad, a.shape[self.axis], self.axis),
                     a.shape,
                     out_shape)
            return [jac]


def matmatmul(m, n):
    return ComputedVariable([m, n],
                            MatrixMatrixOperator(),
                            name='matmatmul({},{})'.format(m.name, n.name))

def grad(output, inputs):
    '''

    :param output: Single variable
    :param inputs: List of variables
    :return: Dictionary of gradients
    '''
    contributions = output._get_contributions()
    grads_cache = {}
    grads_cache[output] = np.ones(1)
    for input in inputs:
        input._get_grad(contributions, grads_cache)
    return OrderedDict([(input, grads_cache[input]) for input in inputs])



def test_operator_jacobian(operator, inputs, epsilon=1e-5):
    inputs = [np.asarray(input, dtype=np.float32) for input in inputs]
    analytical = [jac.to_matrix() for jac in operator.get_jacobians(inputs)]
    numerical = [np.zeros_like(jac) for jac in analytical]
    value = operator(inputs)
    diffs = []
    for j, input in enumerate(inputs):
        for i in xrange(input.size):
            inputs_shifted = copy.deepcopy(inputs)
            unraveled_i = np.unravel_index(i, input.shape)
            inputs_shifted[j][unraveled_i] += epsilon
            value_shifted = operator(inputs_shifted)
            numerical[j][:, i] = (value_shifted-value).flatten() / epsilon
        diff = np.abs((analytical[j]-numerical[j])).mean()
        diffs.append(diff)
    return diffs, analytical, numerical


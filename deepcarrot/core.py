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


class MultiplyOperatorOld(Operator):  # create operator later

    def _call(self, (a, b)):
        return a*b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(np.diag(b))
        jacobian_b = Jacobian(np.diag(a))
        return [jacobian_a, jacobian_b]


class MultiplyOperator(Operator):

    def _call(self, (a, b)):
        return a*b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(lambda x: x*b, a.shape, a.shape)
        jacobian_b = Jacobian(lambda x: x*a, b.shape, b.shape)
        return [jacobian_a, jacobian_b]


class AddOperatorOld(Operator):  # create operator later

    def _call(self, (a, b)):
        return a+b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(np.identity(a.size))  # do those actually make sense for other shapes?
        jacobian_b = Jacobian(np.identity(b.size))
        return [jacobian_a, jacobian_b]


class AddOperator(Operator):  # create operator later

    def _call(self, (a, b)):
        return a+b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(lambda grad: grad, a.shape, a.shape)
        jacobian_b = Jacobian(lambda grad: grad, b.shape, b.shape)
        return [jacobian_a, jacobian_b]


class TransposeOperatorOld(Operator):

    def _call(self, (a,)):
        assert len(a.shape) == 2
        return a.T

    def _get_jacobians(self, (a,)):
        jacobian_a = np.zeros((a.T.shape + a.shape))
        for i in xrange(a.shape[0]):
            for j in xrange(a.shape[1]):
                jacobian_a[j, i, i, j] = 1
        jacobian_a = Jacobian(jacobian_a.reshape(a.size, a.size))
        return [jacobian_a]


class TransposeOperator(Operator):

    def _call(self, (a,)):
        assert len(a.shape) == 2
        return a.T

    def _get_jacobians(self, (a,)):
        jacobian_a = Jacobian(lambda grad: grad.T, a.shape, a.T.shape)
        return [jacobian_a]


# more efficient than jacobian is a linear operator (e.g. matmul)
def prod(shape):
    if len(shape) == 0:
        return 1
    return int(np.prod(shape))

class MatrixMatrixOperatorOld(Operator):

    def _call(self, (a, b)):
        assert len(a.shape) == 2 and len(b.shape) == 2
        return a.dot(b)

    def _get_jacobians(self, (a, b)):
        #p[i,j] = sum_k a[i,k] b[k,j]
        # dp[i,j]/da[i',j'] = (i=i', k=j') b[j',j] , for i=i', j, j'
        # dp[i,j]/da[i',j'] = (i=i', k=j') b[j',j] , for i=i', j, j'
        #pour a: jacob[i,j, i',j'] = 1{i'=i} b[j', j]
        #pour b: jacob[i,j, i',j'] = 1{i'=i} b[j', j]
        out_shape = (a.shape[0], b.shape[1])
        jacobian_a = np.zeros((out_shape + a.shape))
        jacobian_b = np.zeros((out_shape + b.shape))
        for i in xrange(a.shape[0]):
            jacobian_a[i, :, i, :] = b.T
        for j in xrange(b.shape[1]):
            jacobian_b[:, j, :, j] = a
        jacobian_a = Jacobian(jacobian_a.reshape((prod(out_shape), a.size)))
        jacobian_b = Jacobian(jacobian_b.reshape((prod(out_shape), b.size)))
        return [jacobian_a, jacobian_b]


class MatrixMatrixOperator(Operator):

    def _call(self, (a, b)):
        assert len(a.shape) == 2 and len(b.shape) == 2
        return a.dot(b)

    def _get_jacobians(self, (a, b)):
        out_shape = (a.shape[0], b.shape[1])
        jacobian_a = Jacobian(lambda grad: grad.dot(b.T), a.shape, out_shape)
        def closure(grad):
            return a.T.dot(grad)
        jacobian_b = Jacobian(closure, b.shape, out_shape)
        #jacobian_b = Jacobian(lambda grad: a.T.dot(grad), b.shape, out_shape)
        return [jacobian_a, jacobian_b]


class ReluOperatorOld(Operator):

    def _call(self, (a,)):
        return np.maximum(a, 0)

    def _get_jacobians(self, (a,)):
        jacobian_a = Jacobian(np.diag((a.flatten()>0).astype(np.float32)))
        return [jacobian_a]


class ReluOperator(Operator):

    def _call(self, (a,)):
        return np.maximum(a, 0)

    def _get_jacobians(self, (a,)):
        jacobian_a = Jacobian(lambda grad: grad * (a > 0).astype(np.float32), a.shape, a.shape)
        return [jacobian_a]


class SumOperatorOld(Operator):

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
            return [Jacobian(np.ones((1, a.size)))]
        else:
            new_shape = list(a.shape)
            del new_shape[self.axis]
            np.zeros((a.shape[0]))
            jac = np.zeros((np.prod(a.shape), np.prod(new_shape)))
            before_shape = int(np.prod(a.shape[:self.axis]))
            middle_shape = a.shape[self.axis]
            after_shape = int(np.prod(a.shape[self.axis+1:]))
            jac = np.zeros((before_shape, after_shape,
                            before_shape, middle_shape, after_shape))
            for i in xrange(before_shape):
                for j in xrange(after_shape):
                    jac[i, j, i, :, j] = 1
            jac = Jacobian(jac.reshape(before_shape*after_shape, before_shape*middle_shape*after_shape))
            return [jac]


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


import numpy as np
from collections import OrderedDict
import copy
import numbers


DEFAULT_VAR = '<var>'
FloatType = np.float64


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
        '''
        Use this to chain operators on a variable.

        If f is a string, then set name in-place.
        '''
        if isinstance(f, str):
            self.name = f
            return self
        else:  # chain operation
            return f(self)


    def relu(self):
        return ComputedVariable([self],
                                ReluOperator(),
                                name='Relu({})'.format(self.name))

    def sum(self, axis=None):
        return ComputedVariable([self],
                                SumOperator(axis),
                                name='Sum({})'.format(self.name))

    def logsoftmax(self, axis):
        return ComputedVariable([self],
                                LogSoftmaxOperator(axis),
                                name='LogSoftmax({})'.format(self.name))

    def argmax(self, axis=None):
        return ComputedVariable([self],
                                ArgmaxOperator(axis),
                                name='Argmax({})'.format(self.name))


class LeafVariable(AbstractVariable):

    def __init__(self, data, name=DEFAULT_VAR):
        data = np.asarray(data)
        #assert isinstance(data.dtype, numbers.Number), 'Input to LeafVariable can only be iterables of float or int'
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
        inputs = [np.asarray(input, dtype=FloatType) for input in inputs]
        return self._call(inputs)

    def get_jacobians(self, inputs):
        inputs = [np.asarray(input, dtype=FloatType) for input in inputs]
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
        self.in_shape = tuple(in_shape)  # list will break stuff such as indexing
        self.out_shape = tuple(out_shape)

    def jacobian_shape(self):
        return (prod(self.out_shape), prod(self.in_shape))

    def to_matrix(self):
        '''
        This is for testing. Don't use it for computation, the current implementation is slooooow.

        It represents the Jacobian explicitly as a 2D np.ndarray of shape (output_size, input_size).
        Warning: self.function represents the transpose of the JAcobian.
        :return: A jacobian matrix of shape jacobian_shape
        '''
        if isinstance(self.function, np.ndarray):
            return self.function
        else:
            #possible_grads = np.identity(self.jacobian_shape()[0])
            #inputs = []
            #for grad in possible_grads:
            #    input = self.function(grad.reshape(self.out_shape)).flatten()
            #    inputs.append(input)
            #return np.asarray(inputs)  # Need to transpose because function represents transpose of Jacobian
            jacobian = np.zeros(self.out_shape + self.in_shape)
            for i in xrange(prod(self.out_shape)):
                unravel_i = np.unravel_index(i, self.out_shape)
                grad = np.zeros(self.out_shape)
                grad[unravel_i] = 1.
                back = self.function(grad)
                for j in xrange(prod(self.in_shape)):
                    unravel_j = np.unravel_index(j, self.in_shape)
                    jacobian[unravel_i + unravel_j] = back[unravel_j]
            return jacobian  # Need to transpose because function represents transpose of Jacobian

    def backward(self, grad):
        '''
        Left-Multiply a gradient by the transpose of Jacobian
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
        assert a.shape == b.shape, 'No broadcasting for now'
        return a*b

    def _get_jacobians(self, (a, b)):
        jacobian_a = Jacobian(lambda x: x*b, a.shape, a.shape)
        jacobian_b = Jacobian(lambda x: x*a, b.shape, b.shape)
        return [jacobian_a, jacobian_b]


class AddOperator(Operator):

    def _call(self, (a, b)):
        return a+b

    def _get_jacobians(self, (a, b)):
        # Need to detect broadcasting
        def get_closure(dst):
            def closure(grad):
                # Sum up contributions of grad until same shape as b
                while len(grad.shape) > len(dst.shape):
                    grad = grad.sum(axis=0)
                # Sum up when b dimension is singleton
                for i in xrange(len(dst.shape)):
                    if dst.shape[i] < grad.shape[i]:  # was broadcasted
                        grad = grad.sum(axis=i, keepdims=True)
                return grad
            return closure
        # Get output dimension
        out = a+b
        jacobian_a = Jacobian(get_closure(a), a.shape, out.shape)
        jacobian_b = Jacobian(get_closure(b), b.shape, out.shape)
        return [jacobian_a, jacobian_b]


class NLLLossOperator(Operator):
    '''
    Use between LogSoftmax and ground truth integer indices.

    Output is first argument, target is second argument.
    Output: 2D float tensor of shape (batch, classes)
    Ground Truth: 1D integer tensor of shape (batch)
    '''
    def __init__(self, average=True):
        self.average = average

    def __call__(self, inputs):
        output = np.asarray(inputs[0], dtype=FloatType)
        target = np.asarray(inputs[1], dtype=np.int64)
        return self._call([output, target])

    def get_jacobians(self, inputs):
        output = np.asarray(inputs[0], dtype=FloatType)
        target = np.asarray(inputs[1], dtype=np.int64)
        return self._get_jacobians([output, target])

    def _call(self, (output, target)):
        loss = - np.sum(output[range(len(output)), target])
        if self.average:
            loss /= float(len(output))
        return loss

    def _get_jacobians(self, (output, target)):

        def closure(grad):
            assert grad.shape == (), 'Gradient must be scalar'
            out = np.zeros_like(output)
            out[range(len(output)), target] = -1.
            if self.average:
                out /= float(len(output))
            return out * grad

        jacobian_output = Jacobian(closure, output.shape, ())
        return [jacobian_output, None]


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
        jacobian_a = Jacobian(lambda grad: grad * (a > 0).astype(FloatType), a.shape, a.shape)
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
            def closure(grad):
                assert grad.shape == (), 'gradient must be scalar'
                return grad * np.ones_like(a)

            return [Jacobian(closure, a.shape, ())]
            #return [Jacobian(lambda grad: grad * np.ones_like(a), a.shape, ())]
        else:
            out_shape = list(a.shape)
            del out_shape[self.axis]

            def closure(grad):
                grad = np.expand_dims(grad, self.axis)
                return np.repeat(grad, a.shape[self.axis], self.axis)

            jac = Jacobian(closure, a.shape, out_shape)
            return [jac]


class ArgmaxOperator(Operator):

    def __init__(self, axis=None):
        '''
        Sum and reduce, either over one or all axes.
        :param axis:
        '''
        self.axis = axis

    def _call(self, (a,)):
        return a.argmax(axis=self.axis)

    def _get_jacobians(self, (a,)):
        raise Exception('Argmax is not differentiable.')


class LogSoftmaxOperator(Operator):

    def __init__(self, axis):
        '''
        Softmax reduce over an axis.
        '''
        self.axis = axis

    def _call(self, (a,)):
        # Get maximum and subtract it
        a_max = np.max(a, self.axis, keepdims=True)
        a_adjusted = a - a_max  # will broadcast
        exp = np.exp(a_adjusted)
        Z = exp.sum(axis=self.axis, keepdims=True)
        log_softmax = a_adjusted - np.log(Z)
        return log_softmax

    def softmax(self, (a,)):
        a_max = np.max(a, self.axis, keepdims=True)
        a_adjusted = a - a_max  # will broadcast
        exp = np.exp(a_adjusted)
        Z = exp.sum(axis=self.axis, keepdims=True)
        return exp / Z

    # log sum exp(u) = log sum exp(u[i]-umax)exp(umax) = umax * log sum expu[i]-umax)

    def _get_jacobians(self, (a,)):
        log_softmax = self([a])
        softmax = self.softmax([a])

        def closure(grad):
            # wrong! that was for transpose
            # return grad - np.sum(grad*softmax, axis=self.axis, keepdims=True)
            return grad - softmax * np.sum(grad, axis=self.axis, keepdims=True)

        jac = Jacobian(closure, a.shape, a.shape)
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
            def closure(grad):
                assert grad.shape == (), 'gradient must be scalar'
                return grad * np.ones_like(a)

            return [Jacobian(closure, a.shape, ())]
            #return [Jacobian(lambda grad: grad * np.ones_like(a), a.shape, ())]
        else:
            out_shape = list(a.shape)
            del out_shape[self.axis]

            def closure(grad):
                grad = np.expand_dims(grad, self.axis)
                return np.repeat(grad, a.shape[self.axis], self.axis)

            jac = Jacobian(closure, a.shape, out_shape)
            return [jac]


class AccuracyOperator(Operator):

    def _call(self, (a, b)):
        return (a==b).astype(float).mean()

    def _get_jacobians(self, (a,)):
        raise Exception('Not differentiable')


def nll(output, target):
    return ComputedVariable([output, target],
                            NLLLossOperator(),
                            name='nll({},{})'.format(output.name, target.name))

def accuracy(output, target):
    return ComputedVariable([output, target],
                            AccuracyOperator(),
                            name='accuracy({},{})'.format(output.name, target.name))

def relu(m):
    return m.relu()


def logsoftmax(m, axis=1):
    return m.logsoftmax(axis)

def argmax(m, axis=1):
    return m.argmax(axis)

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
    grads_cache[output] = np.asarray(1.)  # propagate scalar of shape ()
    for input in inputs:
        input._get_grad(contributions, grads_cache)
    # Filter out intermediate gradients
    return OrderedDict([(input, grads_cache[input]) for input in inputs])



def test_operator_jacobian(operator, inputs, epsilon=1e-5):
    inputs = [np.asarray(input, dtype=FloatType) for input in inputs]
    analytical = [jac.to_matrix() for jac in operator.get_jacobians(inputs)]
    numerical = [np.zeros_like(jac) for jac in analytical]
    value = operator(inputs)
    diffs = []
    for input_idx, input in enumerate(inputs):
        for j in xrange(input.size):
            inputs_shifted = copy.deepcopy(inputs)
            unraveled_j = np.unravel_index(j, input.shape)
            inputs_shifted[input_idx][unraveled_j] += epsilon
            value_shifted = operator(inputs_shifted)
            grad = (value_shifted - value) / epsilon
            for i in xrange(grad.size):
                # problems with shape () ? i think so
                unraveled_i = np.unravel_index(i, grad.shape)
                numerical[input_idx][unraveled_i + unraveled_j] = grad[unraveled_i]
            #numerical[input_idx][:, i] = (value_shifted-value).flatten() / epsilon  # fill in columns first
        diff = np.abs((analytical[input_idx]-numerical[input_idx])).mean()
        diffs.append(diff)
    return diffs, analytical, numerical


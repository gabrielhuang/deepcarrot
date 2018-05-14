import deepcarrot as dc
import numpy as np


def test_operator(operator, inputs, verbose=True, *args, **kwargs):
    diffs, analytical, numerical = dc.test_operator_jacobian(operator, inputs, *args, **kwargs)
    print '\n\nTEST for Operator', operator
    if verbose:
        print '\n\tInputs', inputs
        print '\n\tAnalytical', analytical
        print '\n\tNumerical', numerical
    print '\n\t->Diffs', np.mean(diffs)
    return diffs


verbose = False
test_operator(dc.AddOperator(), ([1,2],[3,4]), verbose)
test_operator(dc.MultiplyOperator(), ([1,2],[3,4]), verbose)
test_operator(dc.SumOperator(), ([1,2,3,4],), verbose)
test_operator(dc.SumOperator(axis=0), ([[1,2,3,4],[5,6,7,8]],), verbose)
test_operator(dc.ReluOperator(), ([[-1, 1, -2, 2]],), verbose)
test_operator(dc.MatrixMatrixOperator(), ([[1,2],[3,4]], np.identity(2)), verbose)


# Define 3 layers
layer = dc.layers.Dense(5, 3)
relu = dc.layers.Relu()
logsoftmax = dc.layers.LogSoftmax()

# Input and output
x = dc.Variable([np.arange(5), np.arange(5,10)], name='x')
target = dc.Variable([2, 1], name='target')
output = x >> layer >> relu >> logsoftmax >> 'output'
# layers can also be applied using classic composition

loss = dc.nll(output, target)

print x
print target
print output
print loss

grad = dc.grad(loss, layer.parameters())
print 'gradients', grad

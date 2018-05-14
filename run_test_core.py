import deepcarrot as dc
import numpy as np


def test_operator(operator, inputs, verbose=True, *args, **kwargs):
    diffs, analytical, numerical = dc.test_operator_jacobian(operator, inputs, *args, **kwargs)
    if verbose:
        print '\n\nTEST for Operator', operator
        print '\n\tInputs', inputs
        print '\n\tAnalytical', analytical
        print '\n\tNumerical', numerical
        print '\n\t->Diffs', np.mean(diffs)
    return diffs


verbose = True
test_operator(dc.AddOperator(), ([1,2],[3,4]), verbose)
test_operator(dc.MultiplyOperator(), ([1,2],[3,4]), verbose)
test_operator(dc.SumOperator(), ([1,2,3,4],), verbose)
test_operator(dc.SumOperator(axis=0), ([[1,2,3,4],[5,6,7,8]],), verbose)
test_operator(dc.ReluOperator(), ([[-1, 1, -2, 2]],), verbose)
test_operator(dc.MatrixMatrixOperator(), ([[1,2],[3,4]], np.identity(2)), verbose)


layer = dc.layers.Dense(5, 3)
relu = dc.layers.Relu()
x = dc.Variable([np.arange(5)], name='x')
print 'x', x
print 'layer(x)', layer(x)
print layer(x).relu()
y = x >> layer >> relu
y.name = 'y'

dc.grad(y.sum(), layer.parameters())

# Other syntax for chaining layers
print x >> layer >> relu
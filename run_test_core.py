import deepcarrot as dc
import numpy as np

#
# weight = np.asarray([[1,2],[3,4]])
# bias = np.asarray([5,6])
# x = np.asarray([2,2])
# y = np.asarray([1,3])
#
#
# weight = dc.Variable(weight, name='weight')
# bias = dc.Variable(bias, name='bias')
# x = dc.Variable(x, name='x')
# y = dc.Variable(y, name='y')
#
# print weight
# print bias
# print x
# print y
#
# z = x*x+y
# z.name = 'z'
#
# print 'z', z
#
# grads = dc.grad(z, [x, y])
#
# print 'grads', grads
#
#
# u = dc.matmul(weight, x) + bias
# u.name = 'u'
# print u
# grads = dc.grad(u, [weight, bias])
# print grads

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
test_operator(dc.MatrixMatrixOperator(), (np.random.uniform(size=(2,2)), np.random.uniform(size=(2,2))))


layer = dc.layers.Linear(5, 4)
relu = dc.layers.Relu()
x = dc.Variable([np.arange(5)])
print 'x', x
print 'layer(x)', layer(x)
print layer(x).relu()
dc.grad(layer(x).relu().sum(), layer.parameters())

print x >> layer >> relu
def sgd(parameters, grads, lr):
    for p in parameters:
        grad = grads[p]
        p.data -= grad * lr

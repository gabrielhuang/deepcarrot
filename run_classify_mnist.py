import torch
from torchvision import datasets, transforms
import deepcarrot as dc

batch_size = 32
log_interval = 1


def make_infinite(iterable):
    while True:
        for data, label in iterable:
            data = data.numpy()
            label = label.numpy()
            yield data, label

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)
train_iter = make_infinite(train_loader)
test_iter = make_infinite(test_loader)


class Network(object):
    def __init__(self, hidden):
        self.dense = [
            dc.layers.Dense(28*28, hidden),
            dc.layers.Dense(hidden, 10)]

    def __call__(self, x):
        return (x
                >> self.dense[0]
                >> dc.relu
                >> self.dense[1]
                >> dc.logsoftmax)

    def parameters(self):
        return sum([d.parameters() for d in self.dense], [])


net = Network(hidden=64)
parameters = net.parameters()

for iteration, (data, target) in enumerate(train_iter):

    data = dc.Variable(data.reshape(len(data), -1)) >> 'data'
    target = dc.Variable(target) >> 'target'

    # Forward
    output = net(data) >> 'output'

    # Loss
    loss = dc.nll(output, target)

    # Backward
    grad = dc.grad(loss, parameters)

    # Update
    dc.optim.sgd(parameters, grad, lr=0.1)


    # Back to numpy
    if iteration % log_interval == 0:
        print 'Iteration', iteration
        print 'Training loss', loss

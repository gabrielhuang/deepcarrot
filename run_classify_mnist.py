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
            dc.layers.Dense(hidden, hidden),
            dc.layers.Dense(hidden, 10)]

    def __call__(self, x):
        return (x
                >> self.dense[0]
                >> dc.relu
                >> self.dense[1]
                >> dc.relu
                >> self.dense[2]
                >> dc.logsoftmax)

    def parameters(self):
        return sum([d.parameters() for d in self.dense], [])


net = Network(hidden=64)
parameters = net.parameters()
optimizer = dc.optim.SGD(parameters, lr=0.001, momentum=0.0)
train_avg_loss = 0.
test_avg_loss = 0.
test_avg_accuracy = 0.
smooth = 0.99

for iteration, (data, target) in enumerate(train_iter):

    data = dc.Variable(data.reshape(len(data), -1)) >> 'data'
    target = dc.Variable(target) >> 'target'

    # Forward
    output = net(data) >> 'output'

    # Loss
    train_loss = dc.nll(output, target)

    # Backward
    grad = dc.grad(train_loss, parameters)

    # Update
    optimizer.step(grad)

    # Same thing with test set
    data, target = test_iter.next()
    data = dc.Variable(data.reshape(len(data), -1))
    target = dc.Variable(target)
    output = net(data)
    prediction = dc.argmax(output)
    test_loss = dc.nll(output, target)
    test_accuracy = dc.accuracy(prediction, target)

    # Log
    train_avg_loss = smooth * train_avg_loss + (1 - smooth) * train_loss.data
    test_avg_loss = smooth * test_avg_loss + (1 - smooth) * test_loss.data
    test_avg_accuracy = smooth*test_avg_accuracy + (1-smooth) * test_accuracy.data
    train_unbiased_loss = train_avg_loss / (1 - smooth ** (iteration + 1))
    test_unbiased_loss = test_avg_loss / (1 - smooth ** (iteration + 1))
    test_unbiased_accuracy = test_avg_accuracy/ (1-smooth**(iteration+1))

    # Back to numpy
    if iteration % log_interval == 0:
        print 'Iteration', iteration
        print 'Train loss', train_unbiased_loss
        print 'Test  loss', test_unbiased_loss
        print 'Test  accuracy', test_unbiased_accuracy

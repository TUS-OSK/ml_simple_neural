import numpy as np
from simple_neural import Variable, as_var, Layer
import chainer


class Module:
    def __init__(self, *parameters):
        self.parameters = parameters

    def clear_grad(self):
        for p in self.parameters:
            p.clear_grad()

    def iter_params(self):
        for p in self.parameters:
            if isinstance(p, Module):
                yield from p.iter_params()
            else:
                yield p


class Linear(Module, Layer):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = Variable(np.random.randn(out_dim, in_dim).astype(np.float32))
        self.b = Variable(np.random.randn(out_dim).astype(np.float32))
        self.x = None
        self.x_data = None
        self.w_data = None
        super(Linear, self).__init__(self.W, self.b)

    def forward(self, x):
        self.x = as_var(x)
        c = Variable(np.dot(self.W.data, self.x.data) + self.b.data)
        c.grad_fn = self.backward
        self.x_data = self.x.data
        self.w_data = self.W.data
        return c

    def backward(self, x):
        if self.W.grad is None:
            self.W.grad = 0
        if self.b.grad is None:
            self.b.grad = 0
        if self.x.grad is None:
            self.x.grad = 0
        self.x.grad += np.dot(x.grad, self.w_data)
        self.W.grad += np.dot(x.grad[:, None], self.x_data[None])
        self.b.grad += x.grad
        self.x.backward()


class Relu(Layer):
    def forward(self, x):
        self.x = as_var(x)
        self.x_data = x.data
        c = Variable(np.maximum(0, x.data))
        c.grad_fn = self.backward
        return c

    def backward(self, c):
        if self.x.grad is None:
            self.x.grad = 0
        self.x.grad += c.grad * (self.x_data > 0)
        self.x.backward()


def relu(x):
    return Relu()(x)


class Model(Module):
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)
        super(Model, self).__init__(self.l1, self.l2)

    def __call__(self, x):
        x = relu(self.l1(x))
        x = self.l2(x)
        return x


class SGD:
    def __init__(self, lr, parameters):
        self.lr = lr
        self.parameters = list(parameters)

    def update(self):
        for p in self.parameters:
            p.data -= self.lr * (p.grad if p.grad is not None else 0)

    def clear_grad(self):
        for p in self.parameters:
            p.clear_grad()


def softmax(x):
    """
    NaN回避のためにcを全体から引く
    これしても数学的に値は変わらない

    :param x:
    :return:
    """
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    c = np.max(x)
    exp = np.exp(x - c)
    return exp / np.sum(exp)


def categorical_cross_entropy(x, t):
    return -np.log(x[t] + 1e-7)  # NaN回避のためにちょっと足す


class CrossEntropy:
    """
    softmaxしてクロスエントロピーする
    """

    def forward(self, x, t):
        self.x = x
        self.t = np.eye(10)[t]

        sm = softmax(x.data)
        self.sm = sm
        e = categorical_cross_entropy(sm, t)
        c = Variable(e)
        c.grad_fn = self.backward
        return c

    def backward(self, c):
        if self.x.grad is None:
            self.x.grad = 0
        self.x.grad += c.grad * (self.sm - self.t)
        self.x.backward()


def main():
    model = Model()
    opt = SGD(0.001, model.iter_params())
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, 100)

    for ep in range(1, 20):
        for batch in train_iter:
            loss_total = 0
            correct = 0
            for x, y in batch:
                opt.clear_grad()
                out = model(x)
                pred = out.data.argmax()
                if pred == y:
                    correct += 1
                loss = CrossEntropy().forward(out, y)
                loss.backward()
                opt.update()
                loss_total += loss
            mean_loss = loss_total.data / len(batch)
            print("acc:", correct / len(batch), "loss", mean_loss)


if __name__ == '__main__':
    main()

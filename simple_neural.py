import numpy as np
import torch


def as_var(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Layer:
    """
    variableの計算を表す
    """

    def forward(self, *args):
        """
        計算の内容
        :param args:
        :return:
        """
        raise NotImplementedError()

    def backward(self):
        """
        計算の微分
        :return:
        """
        raise NotImplementedError()

    def __call__(self, *args):
        return self.forward(*args)


class Add(Layer):
    """
    加算レイヤー
    """

    def forward(self, a, b):
        """
        普通に足す
        :param a:
        :param b:
        :return:
        """
        self.a = as_var(a)
        self.b = as_var(b)
        self.c = Variable(np.asarray(a.data + b.data))
        self.c.grad_fn = self.backward
        return self.c

    def backward(self):
        """
        勾配は出力の勾配そのまま
        :return:
        """
        if self.a.grad is None:
            self.a.grad = 0
        if self.b.grad is None:
            self.b.grad = 0
        self.a.grad += self.c.grad
        self.b.grad += self.c.grad
        self.a.backward()
        self.b.backward()


class Mul(Layer):
    """
    乗算レイヤー
    """

    def forward(self, a, b):
        """
        普通に掛け算
        :param a:
        :param b:
        :return:
        """
        self.a = as_var(a)
        self.b = as_var(b)
        self.a_data = a.data
        self.b_data = b.data
        self.c = Variable(np.asarray(a.data * b.data))
        self.c.grad_fn = self.backward
        return self.c

    def backward(self):
        """
        勾配は、２つの入力を入れ替える感じで掛ける
        :return:
        """
        if self.a.grad is None:
            self.a.grad = 0
        if self.b.grad is None:
            self.b.grad = 0
        self.a.grad += self.c.grad * self.b_data
        self.b.grad += self.c.grad * self.a_data
        self.a.backward()
        self.b.backward()


class Variable:
    """
    計算に使う変数
    値と勾配を保持している
    """

    def __init__(self, data):
        data = np.asarray(data)
        assert isinstance(data, np.ndarray)
        self.data = data  # 中身
        self.grad = None  # 勾配
        self.grad_fn = None  # 依存する変数の勾配を計算する関数

    def __add__(self, other):
        add = Add()
        c = add(self, other)
        return c

    def __sub__(self, other):
        if not isinstance(other, Variable):
            other = Variable(np.asarray(other))
        add = Add()
        c = add(self, other * -1)
        return c

    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        op = Mul()
        c = op(self, other)
        return c

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Variable(other) - self

    def __str__(self):
        return str(self.data)

    def clear_grad(self):
        self.grad = None
        self.grad_fn = None

    def backward(self):
        if self.grad is None:
            self.grad = 1
        if self.grad_fn is not None:
            self.grad_fn()


def main():
    """
    線形回帰してみる
    y = 10x+5という関数にノイズ混ぜたデータを、
    10a+bというモデルで近似する。
    :return:
    """

    # 学習データ
    xs = np.random.rand(100)
    ys = xs * 10 + 5 + np.random.randn(100)

    # 学習パラメータ
    a = Variable(np.random.rand())
    b = Variable(np.random.rand())

    # 100 epoch回す
    for ep in range(1, 100):
        loss_total = 0
        for x, y in zip(xs, ys):
            a.clear_grad()  # 勾配をクリアして
            b.clear_grad()
            loss = (y - (x * a + b)) * (y - (x * a + b))  # ２乗誤差を計算
            loss_total += loss.data
            loss.backward()  # 誤差伝搬

            a.data -= 0.01 * a.grad  # 計算した誤差で適当に更新
            b.data -= 0.01 * b.grad
        mean_loss = loss_total / len(xs)
        print(f"epoch: {ep}", a, b, mean_loss)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# 单隐层神经网络的简单实现
# 作者：老王 & dyxm

# 矩阵科学计算
import numpy as np


# 双曲函数
def tanh(x):
    return np.tanh(x)


# 双曲函数倒导数：tanh'(x) = 1 - tanh(x)^2
def tanh_deriv(x):
    return 1.0 - np.square(np.tanh(x))


# 逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# 逻辑函数倒数: f'(x) = f(x)*(1 - f(x))
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


# 面向对象
class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 初始化权重w
        self.weights = []
        for i in range(1, len(layers) - 1):
            # np.random.random(n, m) 产生 n 行 m 列的list
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
        # print self.weights

    # BP
    # epochs就是抽样
    def fit(self, X, y, learning_rate=0.3, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            # 随机抽取一个样本对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 预测函数
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

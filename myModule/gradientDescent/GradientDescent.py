# Created by Yuexiong Ding on 2018/5/13.
# 梯度下降
# 
import numpy as np
import matplotlib.pyplot as plt


def batch_gradient_descent(X, y, iteration=1000, alpha=0.01):
    """
    批梯度下降
    :param X: 
    :param y: 
    :param iteration: 训练次数
    :param alpha: 学习率
    :return: 
    """
    count = 0
    # 参数个数
    n = len(X[0])
    theta = np.random.randn(n).reshape(1, -1)

    # 样本个数
    m = len(y)
    # 代价损失
    cost = []
    while count < iteration:
        # 矩阵方式求和
        sum_m = np.dot((np.dot(theta, X.T) - y.reshape(1, -1)), X)
        # 循环方式求和
        # sum_m = np.zeros(n)
        # for i in range(m):
        #     sum_m += (np.dot(theta, X[i].T) - y[i]) * X[i]
        theta -= alpha * (sum_m / m)
        cost.append(loss(X, y, theta))
        count += 1
    return theta, cost


def stochastic_gradient_descent(X, y, iteration=1000, alpha=0.01):
    """
    随机梯度下降
    :param X: 
    :param y: 
    :param iteration: 训练次数
    :param alpha: 学习率
    :return: 
    """
    count = 0
    # 参数个数
    n = len(X[0])
    theta = np.random.randn(n)
    # 样本个数
    m = len(y)
    # 代价损失
    cost = []
    while count < iteration:
        for i in range(m):
            diff = (np.dot(theta, X[i].T) - y[i]) * X[i]
            theta -= alpha * diff
        cost.append(loss(X, y, theta))
        count += 1
    return theta, cost


def mini_batch_gradient_descent(X, y, iteration=1000, alpha=0.01, batch_size=64):
    """
    mini batch 梯度下降
    :param X: 
    :param y: 
    :param iteration: 训练次数
    :param alpha: 学习率
    :param batch_size: 块的大小
    :return: 
    """
    count = 0
    # 参数个数
    n = len(X[0])
    theta = np.random.randn(n)
    # 样本个数
    m = len(y)
    # 代价损失
    cost = []
    if batch_size > m:
        batch_size = m
    while count < iteration:
        for i in range(0, m, batch_size):
            sum_m = np.zeros(n)
            for j in range(i, i + batch_size, 1):
                sum_m += (np.dot(theta, X[j].T) - y[j]) * X[j]
            theta -= alpha * (1.0 / batch_size) * sum_m
        cost.append(loss(X, y, theta))
        count += 1
    return theta, cost


def loss(X, y, theta):
    J = 0
    for i in range(len(X)):
        J += (y[i] - np.dot(theta, X[i].T)) ** 2
    return J


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    y = np.array([2, 4, 6, 8])
    x = np.array([[1], [2], [3], [4]])
    # th, loss = batch_gradient_descent(X, y, iteration=10000, alpha=0.001)
    # th, loss = stochastic_gradient_descent(X, y, iteration=10000, alpha=0.001)
    th, loss = mini_batch_gradient_descent(X, y, iteration=10000, alpha=0.001, batch_size=2)
    th = th.reshape(1, -1)
    print(th)
    plt.plot(x, y, 'g*')
    # plt.scatter(range(len(loss)), loss)
    plt.plot(x, np.dot(X, th.T), 'r')
    plt.show()

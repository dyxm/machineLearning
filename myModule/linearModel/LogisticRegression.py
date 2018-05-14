# Created by Yuexiong Ding on 2018/5/13.
# 逻辑回归
# 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class MyLogisticRegression:
    def __init__(self):
        # 参数
        self.theta = []
        # 类别
        self.classes = []
        pass

    def __sigmoid(self, X):
        return np.array(1.0 / (1.0 + np.exp(-X)))

    def __add_constant(self, X):
        """为特征矩阵增加常数项 b """
        return np.c_[X, np.ones(len(X))]

    def gradient_descent(self, X, y, iteration=10000, alpha=0.01):
        count = 0
        # 参数个数
        n = len(X[0])
        theta = np.random.randn(n).reshape(1, -1)
        # 样本个数
        m = len(y)
        # y = self.sigmoid(y).reshape(1, -1)
        while count < iteration:
            h = self.__sigmoid(np.dot(theta, X.T))
            sum_m = np.dot((h - y.reshape(1, -1)), X)
            theta -= alpha * (sum_m / m)
            count += 1
        return theta.reshape(1, -1)

    def fit(self, X, y, iteration=1000, alpha=1.2):
        """训练模型"""
        X = self.__add_constant(X)
        self.classes = np.unique(y)
        self.theta = self.gradient_descent(X, y, iteration=iteration, alpha=alpha)

    def predict_proba(self, X):
        """预测，返回概率"""
        X = self.__add_constant(X)
        predict_positive = self.__sigmoid(np.dot(self.theta, X.T)).reshape(-1, 1)
        predict_negative = 1 - predict_positive
        return np.c_[predict_negative, predict_positive]

    def predict(self, X):
        """预测，返回 0 或 1 """
        predict_proba = self.predict_proba(X)
        indices = predict_proba.argmax(axis=1)
        return self.classes[indices]


if __name__ == '__main__':
    # 调试专用
    X = np.array([[1, 8], [1.3, 7], [1.5, 7.4], [2, 8.3], [7, 1], [7.2, 1.4], [8, 2], [7.6, 1.8]])
    y_label = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    lr = MyLogisticRegression()
    lr.fit(np.array(X[:, 0]).reshape(-1, 1), y_label, iteration=1000, alpha=1)
    w = lr.theta
    pred = lr.predict(np.array([[1], [2], [9], [8]]))
    print(pred)
    plt.scatter(X[:, 0], X[:, 1])
    plt.plot([0, 9], np.dot(np.array([[0, 1], [9, 1]]), w.T))
    plt.show()
    # print(sigmoid(np.dot(b, a.T)))

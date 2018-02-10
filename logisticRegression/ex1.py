# -*- coding:utf-8 -*-
# 非线性回归分析例子--Logistics回归
# 作者：dyxm

import numpy as np
import random
# 绘图库
import matplotlib.pyplot as plt



# 创建数据 （行数，偏差，方差）
def createData(row, bias, variance):
    x = np.zeros(shape=(row, 2))
    y = np.zeros(shape=row)
    for i in range(row):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# 梯度下降
def gradientDescent(X, y, theta, rate, interators):
    cost = []
    historyTheta = []
    m = np.shape(X)[0]
    for i in range(interators):
        # print theta
        historyTheta.append([theta[0], theta[1]])
        loss = X.dot(theta) - y
        theta = theta - (rate / m) * (X.T.dot(loss))
        c = np.sum(loss ** 2) / (2 * m)
        cost.append(c)
    return theta, cost, historyTheta

x, y = createData(100, 25, 10)
rate = 0.0000005
interators = 10000
theta = [0, -9]
theta, cost, historyTheta = gradientDescent(x, y, theta, rate, interators)
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0][0].plot(cost, color='red')
ax[0][0].set_xlabel('iterators')
ax[0][0].set_ylabel('Loss')
ax[0][0].set_title('Loss Function')
print theta
# print cost

from mpl_toolkits.mplot3d import Axes3D
fig2 = plt.figure()
ax2 = Axes3D(fig2)
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
R = 0
m = np.shape(y)[0]
for i in range(len(x)):
    R += (y[i] - (X * x[i][0] + Y * x[i][1])) ** 2 /(2 * m)
Z = R

ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax2.scatter([t[0] for t in historyTheta], [t[1] for t in historyTheta], cost, color='red')

ax2.set_zlabel('Cost')
ax2.set_xlabel('Theta0')
ax2.set_ylabel('Theta1')

plt.show()
# print y
# -*- coding:utf-8 -*-
# 线性回归分析例子--单元线性回归（鸢尾花）

# 基础库-矩阵运算
import numpy as np
# 数据分析库
import pandas as pd
# 绘图库
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# 可视化库
import seaborn as sns



PATH = r'../dataSet/'
df = pd.read_csv(PATH + 'iris.data', names=['slength', 'swidth', 'plength','pwidth', 'class'])

# 查看各个属性之间的相关性
# print df.corr()
# sns.pairplot(df, hue='class')
# sns.plt.show()

# 提取花瓣长度和宽度
x = df['pwidth'].tolist()
y = df['plength'].tolist()

# fig, ax = plt.subplots(figsize=(12, 12))
# ax.scatter(x, y, color='green')
# ax.set_xlabel('pwidth')
# ax.set_ylabel('plength')
# ax.set_title('Petal Scatterplot')
# plt.show()

# 代价函数
def Cost(x,y,theta):
    J = 0
    for i in range(len(x)):
        J += (y[i] - (theta[0] + theta[1] * x[i]))**2

    return J

# 梯度下降函数
def gradientDescent(x, y, theta, rate, iterators, plt):
    cost = []
    for i in range(iterators):
        t0 = 0
        t1 = 0
        for j in range(len(x)):
            t0 += -(y[j] - (theta[0] + theta[1] * x[j]))
            t1 += -(x[j] * (y[j] - (theta[0] + theta[1] * x[j])))

        theta[0] = theta[0] - rate * t0
        theta[1] = theta[1] - rate * t1
        c = Cost(x, y, theta)
        cost.append(c)
        # print theta
        # print c
        plt.plot(x, [theta[0] + i * theta[1] for i in x])

    return theta, cost



# 参数，初始时 theta0 = 0，theta1 = 0
theta = [0, 0]
# 记录代价变化过程
cost = []
# 学习率
rate = 0.0001
# 迭代次数
iterators = 100

# 散点图与回归线的变化
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0][0].scatter(x, y, color='green')
ax[0][0].set_xlabel('pwidth')
ax[0][0].set_ylabel('plength')
ax[0][0].set_title('Linear Regression')
theta,cost = gradientDescent(x, y, theta, rate, iterators, ax[0][0])

# 代价变化曲线
ax[0][1].plot(cost, color='red')
ax[0][1].set_xlabel('iterators')
ax[0][1].set_ylabel('cost')
ax[0][1].set_title('Loss function')

# 最终结果
# ax[1][0].scatter(x, y, color='green')
# ax[1][0].plot(x, [theta[0] + theta[1] * i for i in x], color='blue')
# ax[1][0].set_xlabel('pwidth')
# ax[1][0].set_ylabel('plength')
# ax[1][0].set_title('Result')


plt.show()
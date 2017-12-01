# -*- coding:utf-8 -*-
# 线性回归分析例子--单元线性回归（鸢尾花）
# 先自写梯度下降算法进行回归分析，在调用专业线性回归分析库进行分析
# 作者：dyxm

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
X = df['pwidth'].tolist()
Y = df['plength'].tolist()

# fig, ax = plt.subplots(figsize=(12, 12))
# ax.scatter(x, y, color='green')
# ax.set_xlabel('pwidth')
# ax.set_ylabel('plength')
# ax.set_title('Petal Scatterplot')
# plt.show()

# 归一化
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return [(i - mean)/std for i in x]

# 代价函数
def Cost(x,y,theta):
    J = 0
    for i in range(len(x)):
        J += (y[i] - (theta[0] + theta[1] * x[i]))**2

    return J

# 梯度下降函数
def gradientDescent(x, y, theta, rate, iterators, plt):
    cost = []
    historyTheta = []
    for i in range(iterators):
        t0 = 0
        t1 = 0
        l = len(x)
        for j in range(len(x)):
            t0 += -(y[j] - (theta[0] + theta[1] * x[j]))
            t1 += -(x[j] * (y[j] - (theta[0] + theta[1] * x[j])))

        # 保存历史theta
        historyTheta.append([theta[0], theta[1]])
        # 更改theta
        theta[0] = theta[0] - rate / l * t0
        theta[1] = theta[1] - rate / l * t1
        # 保存历史代价
        c = Cost(x, y, theta)
        cost.append(c)
        # print theta
        # print c
        plt.plot(x, [theta[0] + i * theta[1] for i in x])

    return theta, cost, historyTheta



# 参数，初始时 theta0 = 0，theta1 = 0
theta = [0, 0]
# 记录代价变化过程
cost = []
# 学习率
rate = 0.07
# 迭代次数
iterators = 1000

# 散点图与回归线的变化
x = normalize(X) # 归一化x
y = normalize(Y) # 归一化y
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0][0].scatter(x, y, color='green')
ax[0][0].set_xlabel('pwidth')
ax[0][0].set_ylabel('plength')
ax[0][0].set_title('Linear Regression')
theta,cost, historyTheta = gradientDescent(x, y, theta, rate, iterators, ax[0][0])
print (theta)
print ([(i - np.mean(X)) / np.std(X) for i in theta])
print (cost)
# 代价变化曲线
ax[0][1].plot(cost, color='red')
ax[0][1].set_xlabel('iterators')
ax[0][1].set_ylabel('cost')
ax[0][1].set_title('Loss function')

# 最终结果
ax[1][0].scatter(x, y, color='green')
ax[1][0].plot(x, [theta[0] + theta[1] * i for i in x], color='blue')
ax[1][0].set_xlabel('pwidth')
ax[1][0].set_ylabel('plength')
ax[1][0].set_title('Result')

# 调用Statsmodels库进行专业的分析
import statsmodels.api as sm
# y = df['plength']
# x = df['pwidth']
X1 = sm.add_constant(x)
results = sm.OLS(y, X1).fit()
print (results.summary())
ax[1][1].plot(x, results.fittedvalues, label='regression line')
ax[1][1].scatter(x, y, color='green')
ax[1][1].set_xlabel('pwidth')
ax[1][1].set_ylabel('plength')
ax[1][1].set_title('Petal Scatterplot')

# 3D图像
from mpl_toolkits.mplot3d import Axes3D
fig2 = plt.figure()
ax2 = Axes3D(fig2)
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
R = 0
for i in range(len(x)):
    R += (y[i] - (X + Y * x[i]))**2
Z = R

ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax2.scatter([t[0] for t in historyTheta], [t[1] for t in historyTheta], cost, color='red')

ax2.set_zlabel('Cost')
ax2.set_xlabel('Theta0')
ax2.set_ylabel('Theta1')

plt.show()
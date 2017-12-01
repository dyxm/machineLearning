# -*- coding:utf-8 -*-
# 线性回归分析例子--多元线性回归
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
df = pd.read_csv(PATH + 'income.data', names=['year', 'GDP', 'savings', 'revenue', 'CPI', 'totalTrade'])
# print df

# 查看各个属性之间的相关性
# print df.corr()
# sns.pairplot(df)
# sns.plt.show()

# # GDP与其他（除年份）属性之间的关系
# # GDP与城乡居民存款
# fig, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax[0][0].scatter(df['savings'], df['GDP'])
# ax[0][0].set_xlabel('savings')
# ax[0][0].set_ylabel('GDP')
# ax[0][0].set_title('GDP And Savings Scatter')
# # GDP与财政收入
# ax[0][1].scatter(df['revenue'], df['GDP'])
# ax[0][1].set_xlabel('revenue')
# ax[0][1].set_ylabel('GDP')
# ax[0][1].set_title('GDP And Revenue Scatter')
# # GDP与居民消费价格指数
# ax[1][0].scatter(df['CPI'], df['GDP'])
# ax[1][0].set_xlabel('CPI')
# ax[1][0].set_ylabel('GDP')
# ax[1][0].set_title('GDP And CPI Scatter')
# # GDP与货物进出口总额
# ax[1][1].scatter(df['totalTrade'], df['GDP'])
# ax[1][1].set_xlabel('totalTrade')
# ax[1][1].set_ylabel('GDP')
# ax[1][1].set_title('GDP And totalTrade Scatter')
# plt.show()

# 读取数据--矩阵形式
x = np.array(df.ix[:, 2: 6]).tolist()
[i.insert(0, 1) for i in x]

X = np.array(x)
Y = np.array(df.ix[:, 1: 2])
# print
# 归一化
def normalize(X):
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - mean[i]) / std[i]
    return X

# 代价函数
def Cost(X, Y, theta):
    # J = (np.sum((X.dot(theta) - Y)**2))
    C = X.dot(theta) - Y
    J = (C.T.dot(C))
    return J[0]


# 梯度下降
def gradientDescent(X, Y, theta, rate, iterators):
    cost = []
    len = np.shape(X)[0]
    for i in range(iterators):
        theta = theta - (rate / len) * (X.T.dot(X.dot(theta) - Y))
        c = Cost(X, Y, theta)
        cost.append(c[0])
        # print c

    return theta, cost

theta = [[0], [0], [0], [0], [0]]
rate = 0.0006
iterators = 10000
cost = []
# 归一化
X = normalize(X)
Y = normalize(Y)

theta, cost = gradientDescent(X, Y, theta, rate, iterators)
print theta
print cost

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax[0][0].scatter(df['year'], df['savings'])
ax[0][0].scatter(df['year'], df['revenue'])
ax[0][0].scatter(df['year'], df['CPI'])
ax[0][0].scatter(df['year'], df['totalTrade'])
ax[0][0].legend(['Savings', 'Revenue', 'CPI', 'TotalTrade'])
ax[0][0].set_xlabel('Year')
ax[0][0].set_ylabel('Money')
ax[0][0].set_title('GDP And Savings Scatter')

ax[0][1].plot(cost, color='red')
ax[0][1].set_xlabel('iterators')
ax[0][1].set_ylabel('Loss')
ax[0][1].set_title('Loss Function')

ax[1][0].scatter(df['year'], Y)
ax[1][0].plot(df['year'], theta[0] + theta[1] * X[:, 1:2] + theta[2] * X[:, 2:3] + theta[3] * X[:, 3:4] + theta[4] * X[:, 4:5], color='green')
ax[1][0].set_xlabel('Year')
ax[1][0].set_ylabel('GDP')
ax[1][0].set_title('GDP And Year Scatter')

# 调用Statsmodels库进行专业的分析
import statsmodels.api as sm
results = sm.OLS(Y, X).fit()
params = results.params
ax[1][1].scatter(df['year'], Y)
ax[1][1].plot(df['year'], params[0] + params[1] * X[:, 1:2] + params[2] * X[:, 2:3] + params[3] * X[:, 3:4] + params[4] * X[:, 4:5], color='green')
ax[1][1].set_xlabel('Year')
ax[1][1].set_ylabel('GDP')
ax[1][1].set_title('GDP And Year Scatter')

plt.show()
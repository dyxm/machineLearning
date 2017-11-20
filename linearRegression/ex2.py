# -*- coding:utf-8 -*-
# 线性回归分析例子--多元线性回归

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
x = np.array(df.ix[:, 2:6]).tolist()
[i.insert(0, 1) for i in x]

X = np.array(x)
Y = np.array(df['GDP'].tolist())

# print X.T
# 代价函数
def Cost(X, Y, theta):
    # J = (np.sum((X.dot(theta) - Y)**2))
    C = X.dot(theta) - Y
    J = (C.T.dot(C))
    return J


# 梯度下降
def gradientDescent(X, Y, theta, rate, iterators):
    cost = []
    for i in range(iterators):
        theta = theta - rate * (X.T.dot(X.dot(theta) - Y))
        c = Cost(X, Y, theta)
        cost.append(c)
        print c

    return theta, cost

theta = [0, 0, 0, 0, 0]
rate = 0.0000000000000006
iterators = 10000
cost = []
theta, cost = gradientDescent(X, Y, theta, rate, iterators)
print theta

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

ax[1][0].scatter(df['year'], df['GDP'])
ax[1][0].plot(df['year'], theta[0] + theta[1] * df['savings'] + theta[2] * df['revenue'] + theta[3] * df['CPI'] + theta[4] * df['totalTrade'], color='green')
ax[1][0].set_xlabel('Year')
ax[1][0].set_ylabel('GDP')
ax[1][0].set_title('GDP And Year Scatter')

# ax[1][1].scatter(df['year'], df['GDP'])
# ax[1][1].plot(df['year'], -8218.578 + 0.338696 * df['savings'] + 2.644429 * df['revenue'] + 95.12859 * df['CPI'] + 0.176135 * df['totalTrade'], color='green')
# ax[1][1].set_xlabel('Year')
# ax[1][1].set_ylabel('GDP')
# ax[1][1].set_title('GDP And Year Scatter')

plt.show()
# -*- coding:utf-8 -*-
import os
import pandas as pd
import requests

# Pandas 数据分析工具

PATH = r'E:/dyxm/projects/PycharmProjects/dataSet/'
#
# r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# with open(PATH + 'iris.data', 'w') as f:
#     f.write(r.text)
# os.chdir(PATH)

df = pd.read_csv(PATH + 'iris.data', names=['slength', 'swidth', 'plength','pwidth', 'class'])

# print df.head()

# 输出单列
# print df['slength']

# 输出前两列和前四行
# print df.ix[:3, :2]

# 列表迭代，和条件搜索
# print df.ix[:3, [x for x in df.columns if 'width' in x]]
# print df[df['class'] == 'Iris-virginica']
# print df[(df['class'] == 'Iris-virginica') & (df['pwidth'] > 2.2)]

# 列出所有可用的唯一类
# print df['class'].unique()

# 统计
# print df.count()
# print df[df['class'] == 'Iris-virginica'].count()

# 重置索引
# print df[df['class'] == 'Iris-virginica'].reset_index(drop = True)

# 输出统计性数据
# print df.describe()
# print df.describe(percentiles=[.20, .40, .80, .90, .95])

# 检查特征之间是否线性相关
# print df.corr()

# 绘图工具 Matplotlib 库
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline
import numpy as np

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 直方图
# fig, ax = plt.subplots(figsize=(6,4))
# ax.hist(df['pwidth'], color='red')
# ax.set_ylabel('Count', fontsize=12)
# ax.set_xlabel('Width',fontsize=12)
# plt.title('Iris pwidth',fontsize=14, y=1.01)
# plt.show()

# 散点图
# fig, ax = plt.subplots(figsize=(6,6))
# ax.scatter(df['pwidth'],df['plength'], color='green')
# ax.set_xlabel('pwidth')
# ax.set_ylabel('plength')
# ax.set_title('Petal Scatterplot')
# plt.show()

# 折线图
# fig, ax = plt.subplots(figsize=(6,6))
# ax.plot(df['plength'], color='blue')
# ax.set_xlabel('bianhao')
# ax.set_ylabel('plength')
# ax.set_title('Petal Length ').
# plt.show()


# seaborn库
import seaborn as sns

# 各属性相关性总览
# print sns.pairplot(df, hue='class')
# sns.plt.show()

# 小提琴图
# fig, ax = plt.subplots(2, 2, figsize=(7,7))
# sns.set(style='white',palette='muted')
# sns.violinplot(x=df['class'], y=df['slength'], ax=ax[0, 0])
# sns.violinplot(x=df['class'], y=df['swidth'], ax=ax[0, 1])
# sns.violinplot(x=df['class'], y=df['plength'], ax=ax[1, 0])
# sns.violinplot(x=df['class'], y=df['pwidth'], ax=ax[1, 1])
# fig.suptitle('Violin Plots', fontsize=16, y=1.03)
# for i in ax.flat:
#     plt.setp(i.get_xticklabels(), rotation=-90)
# fig.tight_layout()
# sns.plt.show()


# Statsmodels库
import statsmodels.api as sm

y = df['plength']
x = df['pwidth']
X = sm.add_constant(x)
results = sm.OLS(y, X).fit()
print (results.summary())
# 画图
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x, results.fittedvalues, label='regression line')
ax.scatter(df['pwidth'], df['plength'], color='green')
ax.set_xlabel('pwidth')
ax.set_ylabel('plength')
ax.set_title('Petal Scatterplot')
plt.show()
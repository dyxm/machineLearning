# coding=utf-8
# 调用sklearn库运行svm分类算法--线性可分
# 实例：二维平面点分类
# 作者：dyxm

import numpy as np
import pylab as pl
from sklearn import svm

# 创建40个数据
np.random.seed()
# 创建 20 个负类和 20 个正类（[2,2]表示均值为 2，方差为 2，使数据服从正太分布）
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# 创建SVM分类器，运用线性核函数
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 画出超平面
w = clf.coef_[0]
# 斜率
k = - w[0] / w[1]
# 产生一个 -5 到 5 的空间
xx = np.linspace(-5, 5)
# 计算y值（超平面）
yy = k * xx - clf.intercept_[0] / w[1]
print (yy)

# 超平面下面的直线
b = clf.support_vectors_[0]
yy_down = k * xx + (b[1] - k * b[0])
# 超平面上面的直线
b = clf.support_vectors_[-1]
yy_up = k * xx + (b[1] - k * b[0])

# 画图
pl.plot(xx, yy, '-')
pl.plot(xx, yy_down, '--')
pl.plot(xx, yy_up, '--')
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=120, color='blue')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.axis('tight')

pl.show()
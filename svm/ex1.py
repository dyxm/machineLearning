# coding=utf-8
# 调用sklearn库运行svm分类算法--简单用法
# 作者：dyxm

from sklearn import svm
x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]

clf = svm.SVC(kernel='linear')
clf.fit(x, y)
print (clf)
# 支持向量点
print (clf.support_vectors_)
# 支持向量点在数据集的下标索引
print (clf.support_)
# 两个类别的支持向量的个数
print (clf.n_support_)
# 预测
print (clf.predict([2, 2]))
# -*- coding: utf-8 -*-
# 调用 NeuralNetwork 实现手写数字的识别
# 作者：老王


import numpy as np
import pylab as pl
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from NeuralNetworks import NeuralNetwork

# 加载数据集
digits = load_digits()
# 64列
X = digits.data
y = digits.target
# 标准化
X -= X.min()
X /= X.max()
# 显示图片
pl.gray()
print (len(digits.images))
pl.matshow(digits.images[1])
pl.show()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print "start fitting"

nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print confusion_matrix(y_test, predictions)
print classification_report(y_test, predictions)

# coding=utf-8
# 调用sklearn库运行svm分类算法---线性不可分
# 实例：人脸识别
# 作者：dyxm

from time import time
import matplotlib.pyplot as pl
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


# ###################################### 初始化数据 ############################################
# 加载人脸库
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
# 样本数
X = lfw_people.data
# 样本特征值的个数
n_features = X.shape[1]
# 样本标记
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print ('Total dataset size:')
print ('n_samples: %d' % n_samples)
print ('n_features: %d' % n_features)
print ('n_classes: %d' % n_classes)

# 分配训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# ############################################################################################


# ###################################### 降维处理 ############################################
n_components = 150
print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print "done in %0.3fs" % (time() - t0)
# 提取影响最大的特征值
eigenfaces = pca.components_.reshape((n_components, h, w))
print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.3fs" % (time() - t0)
# ###########################################################################################


# #################################### 创建分类器 ############################################
print "Fitting the classifier to the training set"
t0 = time()
# c 为惩罚项
# gamma 为多少比例的特征值将被使用
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# 核函数‘rbf’为径向基内核函数
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_
# #########################################################################################


# ############################ 用训练出的模型进行预测 ######################################
print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0)
print classification_report(y_test, y_pred, target_names=target_names)
print confusion_matrix(y_test, y_pred, labels=range(n_classes))
# #########################################################################################


# 画图
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

# 设置标题
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)



prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
pl.show()

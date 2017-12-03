# encoding=utf-8
# 调用sklearn库运行 k邻近 分类算法
# 作者：dyxm

from sklearn import neighbors
from sklearn import datasets

# 分类器
knn = neighbors.KNeighborsClassifier()

# 加载数据集
iris = datasets.load_iris()
# print (iris)
 
# 建立模型
knn.fit(iris.data, iris.target)

# 构建测试数据
newX = [0.1, 0.2, 0.3, 0.4]
# 调用模型预测
predictedLabel = knn.predict(newX)
# 类别编号
print (predictedLabel)
# 类别名
print ( iris.target_names[predictedLabel])
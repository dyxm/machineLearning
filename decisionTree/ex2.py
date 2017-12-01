# coding=utf-8
# 决策树--调用sklearn的相关库
# 作者：dyxm

from sklearn.feature_extraction import DictVectorizer
import csv

from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
allElectronicsData = open(r'book1.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()
print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)
print(labelList)

#
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print ("dummyX:" + str(dummyX))

#
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

#
clf = tree.DecisionTreeClassifier(criterion='entropy') # 默认用基尼指数“card”
clf = clf.fit(dummyX, dummyY)
print ("clf:"+str(clf))

# 将树写进文件
with open("decisionTree.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 用模型进行预测
newX = dummyX[0, :]
newX[0] = 1
newX[2] = 0
print (newX)
print ('预测：' + str(clf.predict(newX)))
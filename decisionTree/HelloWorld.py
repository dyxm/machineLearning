
from sklearn.feature_extraction import DictVectorizer
import csv

from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
allElectronicsData = open(r'E:\book1.csv', 'rb')
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


vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print ("dummyX:" + str(dummyX))
print (vec.get_feature_names())
print("labelList:" +str(labelList))
lb = preprocessing.LabelBinarizer()

dummyY = lb.fit_transform(labelList)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print ("clf:"+str(clf))

with open("d.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)









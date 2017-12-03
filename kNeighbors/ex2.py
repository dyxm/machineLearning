# coding=utf-8
# K 近邻分类算法--代码实现
# 作者：dyxm

import csv
import random
import math
import operator


# 读取数据集
def loadDataSet(fileName, scale, traningSet=[], testSet=[]):
    with open(fileName, 'rb') as csvFile:
        lines = csv.reader(csvFile)
        dataSet = list(lines)
        for x in range(len(dataSet)):
            for y in range(len(dataSet[x]) - 1):
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < scale:
                traningSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])
    return traningSet, testSet


# 计算欧氏距离
def euclideanDistance(sample_1, sample_2, dimension):
    distance = 0
    for x in range(dimension):
        distance += pow(sample_1[x] - sample_2[x], 2)
    return math.sqrt(distance)


# 取最近的k个邻居
def getNeighbors(dataSet, sample, k):
    distances = []
    dimension = len(sample) - 1
    for x in range(len(dataSet)):
        dist = euclideanDistance(sample, dataSet[x], dimension)
        distances.append([dataSet[x], dist])
    distances.sort(key=operator.itemgetter(1))  # 比较第 1 个属性值进行排序
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# 统计k个邻居的类别并输出占多数的类
def classify(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        category = neighbors[x][-1]
        if category in classVotes:
            classVotes[category] += 1
        else:
            classVotes[category] = 1
    sortVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortVotes[0][0]


# 测试函数(返回准确率)
def test(testSet, predictions):
    correctCount = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correctCount += 1
    return (correctCount / float(len(testSet))) * 100.0


# main函数
def main():
    # 训练集
    traningSet = []
    # 测试集
    testSet = []
    # 训练集与测试集的比例
    scale = 0.67
    traningSet, testSet = loadDataSet(r'../dataSet/iris.data', scale)
    print ('训练集：' + repr(len(traningSet)))
    print ('测试集：' + repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(traningSet, testSet[x], k)
        category = classify(neighbors)
        predictions.append(category)
        print ('testSampleCategory：' + testSet[x][-1] + '    predictCategory：' + category)
    accuracy = test(testSet, predictions)
    print ('准确率为：' + repr(accuracy) + '%   ')


# 运行main函数
main()

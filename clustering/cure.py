# Created by [Yuexiong Ding] at 2018/4/19
# cure-层次聚类
#

import numpy as np


class Entity:
    def __init__(self, features):
        self.features = np.array(features)
        self.reps = np.array([features])


class Cure:
    entity = []

    def __init__(self, entity):
        self.entity = entity

    def distant(self, reps1, reps2):
        """计算距离--欧式距离"""
        # 无穷大
        min = float("inf")
        for i in range(len(reps1)):
            for j in range(len(reps2)):
                # 欧式距离
                dis = np.sum((reps1[i] - reps2[j]) ** 2)
                if dis < min:
                    min = dis
        return min

    def find_best_cp(self, entity):
        """找出两个最近的节点"""
        min = float("inf")
        best_i = 0
        best_j = 1
        len = len(entity)
        for i in range(len):
            for j in range(i + 1, len):
                dis = self.distant(entity[i].reps, entity[j].reps) 
                if dis < min:
                    min = dis
                    best_i = i
                    best_j = j
        return best_i, best_j

    def combine_best_cp(self, index1, index2):
        entity = self.entity



def main():
    a = [1, 2, 3]
    b = np.array([a])
    print (len(a))


if __name__ == '__main__':
    main()
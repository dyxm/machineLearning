# Created by Yuexiong Ding on 2018/5/12.
# 蘑菇数据模块
# 

import pymysql
import pymysql.cursors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Mushroom:
    """
    获取蘑菇数据
    """

    def __init__(self, connection_info):
        self.connection = pymysql.connect(host=connection_info['host'], user=connection_info['user'],
                                          password=connection_info['password'], db=connection_info['db'],
                                          port=connection_info['port'], charset=connection_info['charset'])

    def get_row_data(self):
        """
        从数据库获取原始数据
        :return: label, feature
        """
        data = []
        with self.connection.cursor() as cursor:
            sql_1 = 'select * from mushroom'
            cursor.execute(sql_1)
            for row in cursor.fetchall():
                data.append(list(row))
        data = pd.DataFrame(data)
        # 查看类别比例
        print('“p”类共 %d 条，“e”类共 %d 条' % (len(data[data[0] == 'p']), len(data[data[0] == 'e'])))

        return data.iloc[:, 0], data.iloc[:, 1:]

    def get_data(self, test_size=0.3, random_state=None):
        """
        编码原始数据，并分割训练与测试集，返回之
        :param test_size: 测试集占比例，默认0.3
        :param random_state: 随机种子，可选
        :return: X_train, X_test, y_train, y_test
        """
        row_label, row_feature = self.get_row_data()
        l_enc = LabelEncoder()

        # 对类别进行编码，有毒“p”为1，可食用“e”为0
        y = l_enc.fit_transform(row_label)

        # 对22个特征进行one-hot编码
        X = pd.get_dummies(row_feature)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # print(len(y_train), len(y_test))
        return X_train, X_test, y_train, y_test

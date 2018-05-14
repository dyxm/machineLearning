# Created by Yuexiong Ding on 2018/5/12.
# 毒蘑菇分类
# 
from myModule.dataSource.mushroom import Mushroom
from myModule.linearModel.LogisticRegression import MyLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def use_sklearn_model(X_train, X_test, y_train, y_test):
    """用 sklearn 的 LR 模型进行训练和预测"""

    print('******************** sklearn 的 LR 模型 ********************')
    # 训练LR模型
    print('训练LR模型...')
    sk_lr = LogisticRegression()
    sk_lr.fit(X_train, y_train)

    # 预测
    print('预测...')
    # pred = sk_lr.predict_proba(X_test)
    y_pred = sk_lr.predict(X_test)

    # 结果分析
    print('结果分析...')
    print(classification_report(y_test, y_pred))


def use_my_model(X_train, X_test, y_train, y_test):
    """用自己写的 LR 模型进行训练和预测"""

    print('******************** 自己写的 LR 模型 ********************')
    # 训练LR模型
    print('训练LR模型...')
    my_lr = MyLogisticRegression()
    my_lr.fit(X_train, y_train, iteration=300, alpha=3)

    # 预测
    print('预测...')
    # pred = my_lr.predict_proba(X_test)
    y_pred = my_lr.predict(X_test)

    # 结果分析
    print('结果分析...')
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    print('获取蘑菇数据...')
    # 创建数据库连接
    mushroom = Mushroom({'host': '10.33.30.14', 'user': 'dyx', 'password': 'dyx123',
                         'db': 'public_db', 'port': 3306, 'charset': 'utf8'})
    X_tr, X_te, y_tr, y_te = mushroom.get_data()

    # 调用 sklearn 模型
    use_sklearn_model(X_tr, X_te, y_tr, y_te)

    # 调用自己的模型
    use_my_model(X_tr, X_te, y_tr, y_te)

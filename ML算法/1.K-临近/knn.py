import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from utils import plot_learning_curve
from sklearn.feature_selection import SelectKBest


def knn_classify():
    # 生成已标记的数据集
    # 数据集中心点
    centers = [[-2, 2], [2, 2], [0, 4]]
    # X 为特征矩阵
    # y 为样本标签
    X, y = make_blobs(n_samples=60, centers=centers,
                      random_state=0, cluster_std=0.6)

    # cmap就是指matplotlib.colors.Colormap,一个包含三列矩阵的色彩映射表
    # 使用c和cmap来映射颜色，s为形状的大小
    c = np.array(centers)
    # 绘制数据集的散点图
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
    # 绘制中心点的散点图
    plt.scatter(c[:, 0], c[:, 1], s=100, marker='*', c='black')
    plt.show()

    # 对数据集进行训练和拟合
    k = 5
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # 对测试集进行预测
    X_sample = [[0.5, 2], [0, 2], [0.5, 3]]
    y_sample = clf.predict(X_sample)
    print(y_sample)


def knn_regression():
    n_dots = 1000
    # 生成40行1列的服从“0~5”均匀分布的随机样本
    X = 10 * np.random.rand(n_dots, 1)
    y = np.sin(X).flatten()
    # 生成40行1列的服从“-0.1~0.1”均匀分布的随机误差
    y += 0.2 * np.random.rand(n_dots) - 0.1
    k = 5
    knn = KNeighborsRegressor(k)
    knn.fit(X, y)
    T = np.linspace(0, 10, 1000)[:, np.newaxis]
    y_pred = knn.predict(T)
    plt.scatter(X, y, c='g', label='data', s=100)
    plt.plot(T, y_pred, c='k', label='prediction', lw=4)
    plt.axis('tight')
    plt.title('KNeighborsRegressor (k = %i)' % k)
    plt.show()


def knn_diabetes():
    # 加载数据集
    data = pd.read_csv(r'ML算法\1.K-临近\diabetes.csv')
    X = data.iloc[:, :8]
    Y = data.iloc[:, 8]
    # 将数据划分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # 建立模型
    models = []
    # 普通的KNN算法
    models.append(("KNN", KNeighborsClassifier(n_neighbors=2)))
    # 带权重的KNN算法
    models.append(("KNN with weights", KNeighborsClassifier(
        n_neighbors=2, weights="distance")))
    # 指定半径的KNN算法
    models.append(("Radius Neighbors", RadiusNeighborsClassifier(
        radius=500.0)))
    # 算法评价
    # results = []
    # for name, model in models:
    #     kfold = KFold(n_splits=10)
    #     cv_result = cross_val_score(model, X, Y, cv=kfold)
    #     results.append((name, cv_result))
    # for i in range(len(results)):
    #     print("name: {}; cross val score: {}".format(
    #         results[i][0], results[i][1].mean()))

    # 评价出普通 KNN 效果最好
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, Y_train)
    train_score = knn.score(X_train, Y_train)
    test_score = knn.score(X_test, Y_test)
    print("train score: {}\ntest score: {}".format(train_score, test_score))
    # knn = KNeighborsClassifier(n_neighbors=2)
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # plt.figure(figsize=(10, 6))
    # plot_learning_curve(plt, knn, "Learn Curve for KNN Diabetes",
    #                     X, Y, ylim=(0.0, 1.01), cv=cv)
    # plt.show()
    # 选取相关性最高的两大特征
    selector = SelectKBest(k=2)
    X_new = selector.fit_transform(X, Y)
    print('X_new.shape {}'.format(X_new.shape))
    # 使用选出的两大特征计算哪个模型性能好
    # results = []
    # for name, model in models:
    #     kfold = KFold(n_splits=10)
    #     cv_result = cross_val_score(model, X_new, Y, cv=kfold)
    #     results.append((name, cv_result))
    # for i in range(len(results)):
    #     print("name: {}; cross val score: {}".format(
    #         results[i][0], results[i][1].mean()))
    plt.ylabel("BMI")
    plt.xlabel("Glucose")
    plt.scatter(X_new[Y == 0][:, 0], X_new[Y == 0]
                [:, 1], c='r', s=20, marker='o')  # 画出样本
    plt.scatter(X_new[Y == 1][:, 0], X_new[Y == 1]
                [:, 1], c='g', s=20, marker='^')  # 画出样本
    plt.show()


knn_classify()
# knn_regression()
# knn_diabetes()

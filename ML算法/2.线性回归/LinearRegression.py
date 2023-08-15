import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.figure import SubplotParams
# 多项式拟合模型


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    # 这是一个流水线，先增加多项式阶数，然后再用线性回归算法来拟合数据
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


def Regression_sin():
    n_dots = 200
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
    # 把一个n维向量转换成一个n*1维的矩阵
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # 用2/3/5/10阶多项式来拟合数据集
    degrees = [2, 3, 5, 10]
    results = []
    for d in degrees:
        model = polynomial_model(degree=d)
        model.fit(X, Y)
        train_score = model.score(X, Y)
        mse = mean_squared_error(Y, model.predict(X))
        results.append({"model": model, "degree": d,
                       "score": train_score, "mse": mse})
    for r in results:
        print("degree: {}; train score: {}; mean squared error: {}"
              .format(r["degree"], r["score"], r["mse"]))

    # plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
    # for i, r in enumerate(results):
    #     fig = plt.subplot(2, 2, i+1)
    #     plt.xlim(-8, 8)
    #     plt.title("LinearRegression degree={}".format(r["degree"]))
    #     plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    #     plt.plot(X, r["model"].predict(X), 'r-')
    # plt.show()

    plt.figure(figsize=(12, 6), dpi=200)
    X = np.linspace(-20, 20, 2000).reshape(-1, 1)
    Y = np.sin(X).reshape(-1, 1)
    model_10 = results[3]["model"]
    plt.xlim(-20, 20)
    plt.ylim(-2, 2)
    plt.plot(X, Y, 'b-')
    plt.plot(X, model_10.predict(X), 'r-')
    dot1 = [-2*np.pi, 0]
    dot2 = [2*np.pi, 0]
    plt.scatter(dot1[0], dot1[1], s=50, c='r')
    plt.scatter(dot2[0], dot2[1], s=50, c='r')
    plt.show()


def Regression_boston():
    pass


# Regression_sin()
Regression_boston()

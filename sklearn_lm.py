# -*- coding: utf8 -*-
# !/usr/bin/env python
from numpy import *
from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt


# 图形化显示标准线性回归结果, 包括数据集及它的最佳拟合直线
def standPlot(xArr, yArr, ws):

    xMat = mat(xArr)
    yMat = mat(yArr)
    # 画点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 3].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # 画线, 为了保证直线上的点是按顺序排列, 需先升序排序
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws.T
    ax.plot(xCopy[:, 3], yHat)
    plt.show()

def rssError(yArr, yHat):
    return ((yArr - yHat)**2).sum()

if __name__ == "__main__":

    diabetes = datasets.load_diabetes()
    diabetes_X_train = diabetes.data[:-20]
    diabetes_X_test  = diabetes.data[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test  = diabetes.target[-20:]
    lm = linear_model.LinearRegression()
    lm.fit(diabetes_X_train, diabetes_y_train)
    lmResult = lm.predict(diabetes_X_test)

    # Explained variance score: 1 is perfect prediction
    # and 0 means that there is no linear relationship
    # between X and Y.
    print lm.score(diabetes_X_test, diabetes_y_test)
    print rssError(lmResult, diabetes_y_test)
    standPlot(diabetes_X_test, diabetes_y_test, mat(lm.coef_))
# -*- coding: utf8 -*-
# !/usr/bin/env python
__author__ = 'qing'
from numpy import *
from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt

# 标准回归函数, 按公式计算, 这里没法打公式...
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T

    # 判断行列式是否为 0, 若为0, 则矩阵不可逆
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return

    # .I 表示求逆
    ws = xTx.I * (xMat.T*yMat)
    return ws


# 图形化显示标准线性回归结果, 包括数据集及它的最佳拟合直线
def standPlot(xArr, yArr, ws):
    import matplotlib.pyplot as plt
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    # 画点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 3].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # 画线, 为了保证直线上的点是按顺序排列, 需先升序排序
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    print rssError(yMat, yHat)
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
    ws = standRegres(diabetes_X_train, diabetes_y_train)
    standPlot(diabetes_X_test, diabetes_y_test, ws)

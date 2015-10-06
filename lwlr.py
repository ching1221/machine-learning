__author__ = 'qing'
# -*- coding: utf8 -*-
# !/usr/bin/env python

from numpy import *
from sklearn import datasets
import matplotlib.pyplot as plt


def lwlr(testPoint, xArr, yArr, k):  #这里的k即上文的t, 为了与.T区分开
    xMat = mat(xArr); yMat = mat(yArr).T  #T是转置矩阵
    m = shape(xMat)[0]
    weights = mat(eye((m))) #创建一个单位矩阵
    for j in range(m):                      #根据该点建立权重
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:#先检测行列式是否为0，0则不存在逆
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  #I是矩阵的逆
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# 图形化显示局部加权线性回归结果, 包括数据集及它的最佳拟合直线
def lwlrPlot(xArr, yArr, yHat):
    xMat = mat(xArr)
    srtInd = xMat[:, 3].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    # 画线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 3], yHat[srtInd])
    # 画点
    ax.scatter(xMat[:, 3].flatten().A[0],
               mat(yArr).T[:, 0].flatten().A[0],
               s=2, c='red')
    plt.show()

def rssError(yArr, yHat):
    return ((yArr - yHat)**2).sum()


if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    diabetes_X_train = diabetes.data[:-20]
    diabetes_X_test  = diabetes.data[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test  = diabetes.target[-20:]
    lwlrResult = lwlrTest(diabetes_X_test, diabetes_X_train, diabetes_y_train, 0.148)
    print corrcoef(lwlrResult, diabetes_y_test)
    print rssError(lwlrResult, diabetes_y_test)
    lwlrPlot(diabetes_X_test, diabetes_y_test, lwlrResult)
__author__ = 'qing'
# -*- coding: utf8 -*-
# !/usr/bin/env python

from numpy import *
from sklearn import datasets

def rssError(yArr, yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular,  cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat, 0)   #calc mean then subtract it off
    xVar = var(xMat, 0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :]=ws.T
    return wMat

def crossValidation(xArr, yArr, numVal):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)#test ridge results and store
            errorMat[i, k]=rssError(yEst.T.A, array(testY))
            #print errorMat[i, k]
    meanErrors = mean(errorMat, 0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n", unReg
    print "with constant term: ", -1*sum(multiply(meanX, unReg)) + mean(yMat)

if __name__ == "__main__":
    dia = datasets.load_diabetes()
    retX = dia.data
    retY = dia.target
    crossValidation(retX,  retY,  10)
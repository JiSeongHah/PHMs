import numpy as np

def getMax(x):

    absX = abs(x)

    return np.max(absX)

def getMean(x):

    return np.mean(x)

def getRMS(x):

    sqaured = x*x

    meaned = np.mean(sqaured)

    sqrted = np.sqrt(meaned)

    return sqrted


def getSkewness(x):

    meanX = np.mean(x)

    stdX = np.std(x)

    xMinusMeanCubic = (x-meanX)**3

    return np.mean(xMinusMeanCubic)/(stdX**3)

def getKurtosis(x):
    meanX = np.mean(x)

    stdX = np.std(x)

    xMinusMeanCubic = (x - meanX) ** 4


    return np.mean(xMinusMeanCubic) / (stdX ** 4)

def getCrestfactor(x):

    maxX = getMax(x)
    rmsX = getRMS(x)

    return maxX/rmsX

def getImpulseFactor(x):

    maxX = getMax(x)
    meanX = getMean(x)

    return maxX/meanX

def getShapeFactor(x):

    rmsX = getRMS(x)
    meanX = getMean(x)

    return rmsX/meanX


def getFeatureExtraction(x):

    return np.array([getMax(x),\
           getMean(x),\
           getRMS(x),\
           getSkewness(x),\
           getKurtosis(x),\
           getCrestfactor(x),\
           getImpulseFactor(x),\
           getShapeFactor(x)])





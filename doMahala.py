import numpy as np
import matplotlib.pyplot as plt

def getResult_Mahala(refArr,infArr):

    refMean = np.mean(refArr,axis=0)
    refCovMat = np.cov(refArr[:,0],refArr[:,1])

    distLst = []
    for i in range(len(infArr)):

        distSquare = ((refMean-infArr[i,:]).T)@refCovMat@(refMean-infArr[i,:])
        dist = np.sqrt(distSquare)
        distLst.append(dist)

    return distLst



def plotResult_Mahala(refArr,infArr,saveDir=None):

    distLst = getResult_Mahala(refArr=refArr,infArr=infArr)

    plt.scatter(refArr[:,0],refArr[:,1],facecolors='none',edgecolors='b',s=30)
    plt.scatter(infArr[:, 0], infArr[:, 1], c=distLst,s=100,marker='*')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.colorbar()
    plt.show()




# X = np.random.rand(32,2)
# y = np.random.rand(2,2)
#
# n  = 2
#
#
# doit = plotResult_Mahala(X,y)
# print(doit)
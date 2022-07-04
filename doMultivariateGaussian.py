import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def getResult_MulGau(refArr,infArr):

    refMean = np.mean(refArr,axis=0)
    refCovMat = np.cov(refArr[:, 0], refArr[:, 1])

    distribution = multivariate_normal(mean=refMean,cov=refCovMat)
    probLst = []
    for i in range(len(infArr)):
        eachProb = distribution.pdf(infArr[i,:])
        probLst.append(eachProb)

    return probLst, distribution


def plotResult_MulGau(refArr,infArr,saveDir=None):


    probLst, distribution = getResult_MulGau(refArr=refArr,infArr=infArr)

    xMin = min(np.min(refArr[:,0]),np.min(infArr[:,0])) -0.3
    xMax = max(np.max(refArr[:, 0]), np.max(infArr[:, 0])) +0.3
    yMin = min(np.min(refArr[:, 1]), np.min(infArr[:, 1])) -0.3
    yMax = max(np.max(refArr[:, 1]), np.max(infArr[:, 1])) +0.3

    # baseX = np.linspace(xMin,xMax,100)
    # baseY = np.linspace(yMin,yMax,100)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    baseX, baseY = np.mgrid[xMin:xMax:0.01,yMin:yMax:0.01]

    pos = np.dstack((baseX,baseY))
    flattenPos = pos.reshape(-1,2)
    print(flattenPos.shape)
    ProbDenst = distribution.pdf(flattenPos)

    ax.scatter(flattenPos[:,0],flattenPos[:,1],ProbDenst,s=0.01)
    for i in range(len(infArr)):
        ax.scatter(infArr[i,0],infArr[i,1],distribution.pdf(infArr[i,:]),c='r',s=60)
        ax.text(infArr[i,0],infArr[i,1],distribution.pdf(infArr[i,:]), 'test data '+str(i)+
                ' pd : '+str(round(distribution.pdf(infArr[i,:]),2)))
    ax.view_init(elev=30,azim=120)
    ax.xlabel('1st Principal Component')
    ax.ylabel('2nd Principal Component')
    ax.zlabel('probability density function')
    plt.savefig('/home/a286winteriscoming/resultttt.jpg')
    plt.show()

    return xMin,xMax,yMin,yMax


# x = np.random.rand(32,2)
# y = np.random.rand(2,2)
# print(plotResult_MulGau(x,y))
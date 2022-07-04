import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def getResult_Kmeans(refArr,infArr,cluseterNum):

    kmeansCLUSTER = KMeans(n_clusters=cluseterNum,random_state=0).fit(refArr)
    infResult = kmeansCLUSTER.predict(infArr)

    return infResult,kmeansCLUSTER.labels_


def plotResult_Kmeans(refArr,infArr,clusterNum,saveDir=None):

    infResult,refResult = getResult_Kmeans(refArr=refArr,
                                           infArr=infArr,
                                           cluseterNum=clusterNum)

    plt.scatter(refArr[:,0],refArr[:,1],c=refResult,s=30)
    plt.scatter(infArr[:, 0], infArr[:, 1], c=infResult,s=100,marker='*')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()





# X = np.random.rand(32,2)
# y = np.random.rand(2,2)
#
# n  = 2
#
#
# doit = plotResult_Kmeans(X,y,2)
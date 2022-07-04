import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def getResult_KNN(refArr,infArr,neiNum):

    mok, rest = divmod(len(refArr),neiNum)

    arr4Zero = np.zeros(rest)
    labelrest=  np.array([i for i in range(neiNum)])
    labelrest = np.repeat(labelrest,mok,axis=0)

    refLabel = np.concatenate((arr4Zero,labelrest))

    KNNclassifier = KNeighborsClassifier(n_neighbors=neiNum)

    KNNclassifier.fit(refArr,refLabel)

    predRef = KNNclassifier.predict(refArr)
    prefInf = KNNclassifier.predict(infArr)

    return predRef,prefInf

def plotResult_KNN(refArr,infArr,neiNum,saveDir=None):

    predRef,predInf = getResult_KNN(refArr=refArr,infArr=infArr,neiNum=neiNum)

    # plt.scatter(refArr[:,0],refArr[:,1],facecolors='none',edgecolors=int(predRef),s=30)
    plt.scatter(refArr[:, 0], refArr[:, 1], c= predRef, s=30)
    plt.scatter(infArr[:, 0], infArr[:, 1], c=predInf,s=100,marker='*')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    # plt.colorbar()
    plt.show()










# x = np.array([0,1,2])
# y = np.repeat(x,3,axis=0)
# print(y)
# z = np.zeros(3)
# print(z)


x = np.random.rand(32,2)
y = np.random.rand(4,2)

doit = plotResult_KNN(x,y,3)


import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def getResult_SVM(refArr,infArr):
    mok, rest = divmod(len(refArr), 2)

    if rest != 0:
        arr4Zero = np.zeros(rest)
        labelrest = np.array([i for i in range(2)])
        labelrest = np.repeat(labelrest, mok, axis=0)

        refLabel = np.concatenate((arr4Zero, labelrest))
    else:
        labelrest = np.array([i for i in range(2)])
        refLabel = np.repeat(labelrest, mok, axis=0)

    scaler = StandardScaler()
    scaledRefArr = scaler.fit_transform(refArr)
    scaledInfArr = scaler.transform(infArr)

    classifier = SVC(kernel='sigmoid').fit(refArr,refLabel)

    refResult = classifier.predict(refArr)
    infResult = classifier.predict(infArr)

    return refResult,infResult


def plotResult_SVM(refArr,infArr,saveDir=None):

    refResult,infResult = getResult_SVM(refArr=refArr,infArr=infArr)

    plt.scatter(refArr[:,0],refArr[:,1],c=refResult,s=30)
    plt.scatter(infArr[:, 0], infArr[:, 1], c=infResult,s=100,marker='*')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()



x = np.random.rand(32,2)
y = np.random.rand(3,2)

doit = plotResult_SVM(x,y)
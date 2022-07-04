import numpy as np
import os
from doFeatureExtraction import getFeatureExtraction
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.axes3d import Axes3D
import copy

def loadData(dir):

    return np.array(np.loadtxt(dir,delimiter='\t'))

def appendExtractedFeatureArr(lst,infoLst,channel):

    rawArrDir, idx = infoLst[0],infoLst[1]

    loadedRawArr = loadData(rawArrDir)

    ExtracedFeatureArr = getFeatureExtraction(loadedRawArr[:,channel])

    lst.append([ExtracedFeatureArr,idx])

    print(f'{idx} th thing complete')



def ExceuteFeatureExtractionforAllArr(dir,channel,test):

    fileLst = sorted(os.listdir(dir))
    fileLst = [dir+eachFile for eachFile in fileLst]

    if test== 'ver1':

        totalArr = []
        for idx,eachFileDir in enumerate(fileLst):

            loadedData =loadData(eachFileDir)

            FeatureExtracted = getFeatureExtraction(loadedData[:,channel])

            totalArr.append(FeatureExtracted)
            print(f'{idx}/{len(fileLst)} complete')

        totalArr = np.stack(totalArr)
        return totalArr

    if test == 'ver2':

        totalArr= Manager().list()
        doLst = []
        for idx,eachFileDir in enumerate(fileLst):
            doLst.append([eachFileDir,idx])

        with ProcessPoolExecutor(max_workers=16) as executor:

            for result in executor.map(appendExtractedFeatureArr,repeat(totalArr),doLst,channel):
                pass
        for i in totalArr:
            print(totalArr[0])

        totalArr = sorted(totalArr,key= lambda x : x[1])
        totalArr2 = []
        for i in totalArr:
            totalArr2.append(i[0])
            print(f'appending {i[0]} th')

        TOTALARR = np.stack(totalArr2)

        # meanTotalArr = np.mean(TOTALARR,axis=1)
        # stdTotalARr = np.std(TOTALARR,axis=1)

        # TOTALARR = (TOTALARR-meanTotalArr)/stdTotalARr

        return TOTALARR


def plotEachFeature(arr,saveDir,test):

    lstLen = len(arr)
    count = 0
    xLst = []
    for i in range(lstLen):
        xLst.append(count)
        print(i)
        if i < 42:
            if test == 'test1':
                count +=5
            else:
                count +=10
        else:
            count += 10

    arrMax = arr[:,0]
    arrMean = arr[:,1]
    arrRMS = arr[:,2]
    arrSkewness = arr[:,3]
    arrKurtosis = arr[:,4]
    arrCrestFactor = arr[:,5]
    arrImpulseFactor = arr[:,6]
    arrShapefactor = arr[:,7]


    fig = plt.figure(constrained_layout=True)

    ax1 = fig.add_subplot(4,2,1)
    ax1.plot(xLst,arrMax)
    ax1.set_xlabel('minute')
    ax1.set_ylabel('value')
    ax1.set_title('Max')

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(xLst, arrMean)
    ax2.set_xlabel('minute')
    ax2.set_ylabel('value')
    ax2.set_title('Mean')

    ax3 = fig.add_subplot(4,2, 3)
    ax3.plot(xLst, arrRMS)
    ax3.set_xlabel('minute')
    ax3.set_ylabel('value')
    ax3.set_title('RMS')

    ax4 = fig.add_subplot(4,2, 4)
    ax4.plot(xLst, arrSkewness)
    ax4.set_xlabel('minute')
    ax4.set_ylabel('value')
    ax4.set_title('Skewness')

    ax5 = fig.add_subplot(4,2, 5)
    ax5.plot(xLst, arrKurtosis)
    ax5.set_xlabel('minute')
    ax5.set_ylabel('value')
    ax5.set_title('Kurtosis')

    ax6 = fig.add_subplot(4,2, 6)
    ax6.plot(xLst, arrCrestFactor)
    ax6.set_xlabel('minute')
    ax6.set_ylabel('value')
    ax6.set_title('CrestFactor')

    ax7 = fig.add_subplot(4,2, 7)
    ax7.plot(xLst, arrImpulseFactor)
    ax7.set_xlabel('minute')
    ax7.set_ylabel('value')
    ax7.set_title('ImpulseFactor')

    ax8 = fig.add_subplot(4,2, 8)
    ax8.plot(xLst, arrShapefactor)
    ax8.set_xlabel('minute')
    ax8.set_ylabel('value')
    ax8.set_title('ShapeFactor')

    plt.savefig( saveDir, dpi=200)
    print('saving plot complete!')
    plt.close()

def plotPCA3dScatter(arr,saveDir=None):

    X = arr[:,0]
    print(X.shape)
    Y = arr[:,1]
    print(Y.shape)
    Z = arr[:,2]
    print(Z.shape)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    ax.scatter(X,Y,Z)
    ax.set_xlabel('1st PC')
    ax.set_ylabel('2nd PC')
    ax.set_zlabel('3rd PC')
    plt.savefig( saveDir, dpi=200)


def doPreprocessing4PCA(x):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform((x))

    return X_scaled

def doPCA(x,numComponent):

    x = doPreprocessing4PCA(x)
    xx = copy.deepcopy(x)

    pca = PCA(n_components=numComponent)
    pca.fit(x)
    pcaArr = pca.transform(x)

    pca2 = PCA()
    pca2.fit(xx)


    return pca2,pcaArr


def saveResults(DataDir,
                FESaveDir,
                PCASaveDir,
                csvSaveDir,
                name,
                channel):


    rawArr = ExceuteFeatureExtractionforAllArr(DataDir,channel,test='ver2')

    pcaSolver,pcaArr= doPCA(rawArr,3)

    coeffMatrix = pcaSolver.components_

    plotEachFeature(rawArr, saveDir=FESaveDir +name+'_'+str(channel)+ '_FE.png')
    plotPCA3dScatter(pcaArr, saveDir=PCASaveDir +name+'_'+str(channel)+ '_PCA.png')
    np.savetxt(csvSaveDir+name+'_'+str(channel)+'_coeff.csv',coeffMatrix,delimiter=',')


baseDir = '/home/a286winteriscoming/Downloads/IMS bearing/'
DirLst = ['1st_test','2nd_test','4th_test']

for i in DirLst:
    saveResults(DataDir=baseDir+i+'/',
                FESaveDir=baseDir,
                PCASaveDir=baseDir,
                csvSaveDir=baseDir,
                name=i)












import kmeans
import numpy as np
import readAndWriteDataSet
import getCovMatrix


def pca(channelData, informations, centroidList, clusterAssment, rate=1):
    VT2s = []
    rates = np.array(np.zeros((len(channelData), 2)), dtype=complex)
    rateList = []
    newChannelDataList = []

    # 计算变换矩阵
    for i in range(len(centroidList)):
        U, Sigma, VT = np.linalg.svd(centroidList[i])
        sum = np.sum(Sigma)
        curSum = 0
        index = 0
        if rate <= 1:
            for j in range(len(Sigma)):
                curSum += Sigma[j]
                if rate - (curSum / sum) > 0:
                    index += 1
                else:
                    break
        else:
            index = rate - 1
        VT2 = VT[0:index + 1, :]
        VT2s.append(VT2)

    # 降维
    for i in range(len(channelData)):
        newChannelData = np.dot(channelData[i], np.transpose(VT2s[(int)(clusterAssment[i, 0].real)]))
        newChannelDataList.append(newChannelData)
        index = np.shape(newChannelData)[1]
        rates[i, 0] = index

    newCovMatrixList = getCovMatrix.getCovMatrixList(newChannelDataList)
    newInformations = getCovMatrix.getInformations(newCovMatrixList)[0]

    for i in range(len(channelData)):
        rate2 = newInformations[0][i] / informations[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, VT2s, rateList


def pca_U(channelDataList, informations, centroidList, clusterAssment, newDimension=1):
    newChannelDataList = []
    U2s = []

    rates = np.array(np.zeros((len(channelDataList), 2)), dtype=complex)
    # 为了输出，要把rates放到list中
    rateList = []

    # 计算变换矩阵
    for i in range(len(centroidList)):
        U2 = np.conjugate(centroidList[i][:, 0:newDimension])
        U2s.append(U2)

    # 降维
    for i in range(len(channelDataList)):
        newChannelData = np.dot(channelDataList[i], U2s[(int)(clusterAssment[i, 0].real)])
        newChannelDataList.append(newChannelData)

    newCovMatrixList = getCovMatrix.getCovMatrixList(newChannelDataList)
    newInformation = getCovMatrix.getInformations(newCovMatrixList)[0]

    for i in range(len(channelDataList)):
        rate2 = newInformation[0][i] / informations[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, U2s, rateList


def pca_S(SigmaList, newDimension=1):
    rates = np.array(np.zeros((len(SigmaList), 2)), dtype=complex)
    rateList = []
    # 降维
    for i in range(len(SigmaList)):
        rate2 = sum(SigmaList[i][0:newDimension])
        rates[i, 0] = newDimension
        rates[i, 1] = rate2
    rateList.append(rates)
    return rateList


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    k = 100
    centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
    pca(dataSet, centroids, clusterAssment)

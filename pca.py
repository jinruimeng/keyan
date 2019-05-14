import kmeans
import numpy as np
import readAndWriteDataSet
import getCovMatrix


def pca(channelData, covMatrixList, centroidList, clusterAssment, rate=1):
    # Us = []
    VTs = []
    VT2s = []
    newChannelDataAllList = []
    rates = np.array(np.zeros((len(channelData), 2)), dtype=complex)
    rateList = []
    newChannelDataList = []
    newCovMatrixList = []

    # 计算变换矩阵
    for i in range(len(centroidList)):
        U, Sigma, VT = np.linalg.svd(centroidList[i], full_matrices=0)
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
        # U2 = U[:, 0:index]
        # Us.append(U2)
        VTs.append(VT)
        VT2 = VT[0:index + 1, :]
        VT2s.append(VT2)

    # 降维
    for i in range(len(channelData)):
        # newCovMatrixs.append(
        #     np.dot(np.linalg.inv(Us[(int)(clusterAssment[i, 0])]),
        #            np.dot(covMatrixs[i],
        #                   np.linalg.inv(VTs[(int)(clusterAssment[i, 0])]))))
        # newCovMatrixList.append(
        #     np.dot(VTs[(int)(clusterAssment[i, 0].real)],
        #            np.dot(covMatrixList[i],
        #                   np.transpose(VTs[(int)(clusterAssment[i, 0].real)]))))
        newChannelDataAll = np.dot(channelData[i], np.transpose(VTs[(int)(clusterAssment[i, 0].real)]))
        newChannelDataAllList.append(newChannelDataAll)

        index = np.shape(VT2s[(int)(clusterAssment[i, 0].real)])[0]
        rates[i, 0] = index
        newChannelData = newChannelDataAll[:, 0:index]
        newChannelDataList.append(newChannelData)

    newCovMatrixAllList = getCovMatrix.getCovMatrixList(newChannelDataAllList)
    newInformationAll = getCovMatrix.getInformations(newCovMatrixAllList)

    # for i in range(len(newCovMatrixAllList)):
    #     index = np.shape(VT2s[(int)(clusterAssment[i, 0].real)])[0]
    #     newCovMatrix = newCovMatrixAllList[i][0:index, 0:index]
    #     newCovMatrixList.append(newCovMatrix)
    # newInformation = getCovMatrix.getInformations(newCovMatrixList)

    newCovMatrixList = getCovMatrix.getCovMatrixList(newChannelDataList)
    newInformation = getCovMatrix.getInformations(newCovMatrixList)

    for i in range(len(channelData)):
        rate2 = newInformation[0][i] / newInformationAll[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, VT2s, rateList


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    k = 100
    centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
    pca(dataSet, centroids, clusterAssment)

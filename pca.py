import kmeans
import numpy as np
import readAndWriteDataSet


def pca(channelData, covMatrixList, centroidList, clusterAssment, rate=1):
    # Us = []
    VTs = []
    newChannelData = []
    newCovMatrixList = []

    # 计算变换矩阵
    for i in range(len(centroidList)):
        U, Sigma, VT = np.linalg.svd(centroidList[i])
        sum = np.sum(Sigma)
        curSum = 0
        index = 0
        for j in range(len(Sigma)):
            curSum += Sigma[j]
            if rate - curSum / sum > 0:
                index += 1
            else:
                break
        # U2 = U[:, 0:index]
        # Us.append(U2)
        VT2 = VT[0:index + 1, :]
        VTs.append(VT2)
    # 降维
    for i in range(len(covMatrixList)):
        # newCovMatrixs.append(
        #     np.dot(np.linalg.inv(Us[(int)(clusterAssment[i, 0])]),
        #            np.dot(covMatrixs[i],
        #                   np.linalg.inv(VTs[(int)(clusterAssment[i, 0])]))))
        newCovMatrixList.append(
            np.dot(VTs[(int)(clusterAssment[i, 0].real)],
                   np.dot(covMatrixList[i],
                          np.transpose(VTs[(int)(clusterAssment[i, 0].real)]))))
        newChannelData.append(np.dot(channelData[i], np.transpose(VTs[(int)(clusterAssment[i, 0].real)])))

    return newChannelData, newCovMatrixList


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    k = 100
    centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
    pca(dataSet, centroids, clusterAssment)

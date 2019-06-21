import kmeans
import numpy as np
import readAndWriteDataSet
import tools


def pca(channelData, informations, centroidList, clusterAssment, rate=1):
    U2s = []
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
        U2 = np.transpose(VT[0:index + 1, :])
        U2s.append(U2)

    # 降维
    for i in range(len(channelData)):
        newChannelData = np.dot(channelData[i], U2s[(int)(clusterAssment[i, 0].real)])
        newChannelDataList.append(newChannelData)
        index = np.shape(newChannelData)[1]
        rates[i, 0] = index

    newCovMatrixList = tools.getCovMatrixList(newChannelDataList)
    newInformations = tools.getInformations(newCovMatrixList)[0]

    for i in range(len(channelData)):
        rate2 = newInformations[0][i] / informations[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, U2s, rateList


def pca_U(channelDataList, informations, centroidList, clusterAssment, newDimension=1):
    newChannelDataList = []
    U2s = []

    rates = np.array(np.zeros((len(channelDataList), 2)), dtype=complex)
    # 为了输出，要把rates放到list中
    rateList = []

    # 计算变换矩阵
    for i in range(len(centroidList)):
        U2 = centroidList[i][:, 0:newDimension]
        U2s.append(U2)

    # 降维
    for i in range(len(channelDataList)):
        newChannelData = np.dot(channelDataList[i], U2s[(int)(clusterAssment[i, 0].real)])
        newChannelDataList.append(newChannelData)

    newCovMatrixList = tools.getCovMatrixList(newChannelDataList)
    newInformation = tools.getInformations(newCovMatrixList)[0]

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

# pca
# input: data-信道数据 newDimension-保留维度数
# output： out-pca处理后的数据
def pca_general(data, newDimension=1):
    try:
        # 如果输入是单个信道，进行以下步骤
        m, n = np.shape(data)
        # 计算协方差矩阵 rowvar=False代表每一列是一个变量
        covMatrix = np.cov(data, rowvar=False)
        # SVD分解协方差矩阵得出变换矩阵
        U = np.transpose(np.linalg.svd(covMatrix)[2])
        return np.dot(data, U[:, 0:newDimension])

    except:
        # 如果输入是多个信道，进行以下步骤
        out = []
        covList = tools.getCovMatrixList(data)
        UList = tools.getInformations(covList)[2]
        for i in range(len(data)):
            out.append(np.dot(data[i], UList[i][:, 0:newDimension]))
        return out


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    k = 100
    centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
    pca(dataSet, centroids, clusterAssment)

import kmeans
import numpy as np
import readAndWriteDataSet
import tools


def pca(channelData, informations, centroidList, clusterAssment, rate=1):
    U2s = []
    rates = np.array(np.zeros((np.shape(channelData)[0], 2)))
    rateList = []
    newChannelDataList = []

    # 计算变换矩阵
    for i in range(np.shape(centroidList)[0]):
        U, Sigma, VT = np.linalg.svd(centroidList[i])
        sum = np.sum(Sigma)
        curSum = 0
        index = 0
        if rate <= 1:
            for j in range(np.shape(Sigma)[0]):
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
    for i in range(np.shape(channelData)[0]):
        newChannelData = np.dot(channelData[i], U2s[(int)(clusterAssment[i, 0].real)])
        newChannelDataList.append(newChannelData)
        index = np.shape(newChannelData)[1]
        rates[i, 0] = index

    newCovMatrixList = tools.getCovMatrixList(newChannelDataList)
    newInformations = tools.getInformations(newCovMatrixList)[0]

    for i in range(np.shape(channelData)[0]):
        rate2 = newInformations[0][i] / informations[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, U2s, rateList


def pca_U(channelDataList, informations, centroidList, clusterAssment, newDimension=1):
    newChannelDataList = []
    U2s = []

    rates = np.array(np.zeros((np.shape(channelDataList)[0], 2)))
    # 为了输出，要把rates放到list中
    rateList = []

    # 计算变换矩阵
    for i in range(np.shape(centroidList)[0]):
        U2 = centroidList[i][:, 0:newDimension]
        U2s.append(U2)

    # 降维
    for i in range(np.shape(channelDataList)[0]):
        newChannelData = np.dot(channelDataList[i], U2s[(int)(clusterAssment[i, 0].real)])
        newChannelDataList.append(newChannelData)

    newCovMatrixList = tools.getCovMatrixList(newChannelDataList)
    newInformation = tools.getInformations(newCovMatrixList)[0]

    for i in range(np.shape(channelDataList)[0]):
        rate2 = newInformation[0][i] / informations[0][i]
        rates[i, 1] = rate2

    rateList.append(rates)
    return newChannelDataList, newCovMatrixList, U2s, rateList


def pca_S(SigmaList, newDimension=1):
    rates = np.array(np.zeros((np.shape(SigmaList)[0], 2)))
    rateList = []
    # 降维
    for i in range(np.shape(SigmaList)[0]):
        rate2 = sum(SigmaList[i][0:newDimension])
        rates[i, 0] = newDimension
        rates[i, 1] = rate2
    rateList.append(rates)
    return rateList


# pca
# input: data-信道数据 newDimension-保留维度数
# output： out-pca处理后的数据
def pca_general(data, iRate):
    try:
        # 如果输入是单个信道，进行以下步骤
        # 计算协方差矩阵 rowvar=False代表每一列是一个变量
        covMatrix = np.cov(data, rowvar=False)
        # SVD分解协方差矩阵得出变换矩阵
        U, Sigma, VT = np.linalg.svd(covMatrix)
        return np.dot(data, np.transpose(VT)[:, 0:iRate]), np.transpose(VT)[:, 0:iRate]

    except:
        print(u'pca_general')
        # 如果输入是列表，进行以下步骤
        out = []
        covList = tools.getCovMatrixList(data)
        informations, SigmaList, UList = tools.getInformations(covList)
        UList2 = []
        for i in range(np.shape(SigmaList)[0]):
            out.append(np.dot(data[i], UList[i][:, 0:iRate]))
            UList2.append(UList[i][:, 0:iRate])
        return out, UList2


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    k = 100
    centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
    pca(dataSet, centroids, clusterAssment)

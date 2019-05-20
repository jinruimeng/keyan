# 支持中文显示
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import readAndWriteDataSet
import multiprocessing
import numpy as np
import getCovMatrix
import kmeans
import pca


def elbow(channelDataAll, low, high, step, a, iRateOrK, type2, type="oushi"):
    '利用SSE选择k'
    SSE_S = []  # 存放每次结果
    SSE_C = []  # 存放每次结果
    SSE_U = []  # 存放每次结果
    ps = multiprocessing.Pool(4)

    # 维度固定iRateOrK == iRate
    if type2 == 0:
        for i in range(low, high + 1, step):
            # SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, i, iRateOrK, type)).get()
            SSE = elbowCore(channelDataAll, a, i, iRateOrK, type)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel("聚类中心数量")

    # 聚类中心数量固定iRateOrK == k
    else:
        for i in range(low, high + 1, step):
            # SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, iRateOrK, i, type)).get()
            SSE = elbowCore(channelDataAll, a, iRateOrK, i, type)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel("保留维度")

    ps.close()
    ps.join()
    X = range(low, high + 1, step)
    plt.ylabel("信息量保留")
    plt.plot(X, SSE_S, 'b-o')
    plt.plot(X, SSE_C, 'r-v')
    plt.plot(X, SSE_U, 'k-s')
    plt.show()
    print("主进程结束！")


# i:聚类中心数量 iRate:维度
def elbowCore(channelDataAll, a, k, iRate, type="oushi"):
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    sub = n >> a
    rates_C = []
    rates_U = []
    rates_S = []

    for g in range(1 << a):
        channelData = []
        for h in range(p):
            channelDataPage = channelDataAll[h]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])

        covMatrixList = getCovMatrix.getCovMatrixList(channelData)
        allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

        # 对协方差进行聚类
        centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, k)
        centroidList = getCovMatrix.matrixToMatrixList(centroids)

        # 分析PCA效果,计算信息量保留程度
        tmpRates = pca.pca(channelData, centroidList, clusterAssment, iRate)[3][0][:, 1]
        rates_C.append(np.mean(tmpRates))

        # 对变换矩阵进行聚类
        UList = []
        SigmaList = []
        for h in range(p):
            U, Sigma, VT = np.linalg.svd(covMatrixList[h], full_matrices=0)
            UList.append(U)
            sumWeight = sum(Sigma)
            for i in range(len(Sigma)):
                Sigma[i] = Sigma[i] / sumWeight
            SigmaList.append(Sigma)
        allU = getCovMatrix.matrixListToMatrix_U(UList)
        weights = getCovMatrix.matrixListToMatrix_U(SigmaList)
        centroids, clusterAssment = kmeans.KMeansOushi_U(allU, k, weights, iRate)
        centroidList = getCovMatrix.matrixToMatrixList_U(centroids)

        # 分析PCA效果,计算信息量保留程度
        tmpRates = pca.pca_U(channelData, centroidList, clusterAssment, iRate)[3][0][:, 1]
        rates_U.append(np.mean(tmpRates))

        # 不聚类，直接PCA
        tmpRates = pca.pca_S(covMatrixList, iRate)[0][:, 1]
        rates_S.append(np.mean(tmpRates))

    rate_C = np.mean(rates_C)
    rate_U = np.mean(rates_U)
    rate_S = np.mean(rates_S)

    return rate_S.real, rate_C.real, rate_U.real


if __name__ == '__main__':
    type = "oushi"
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"
    a = 2

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    # 确定维度，改变聚类中心数量
    iRate = 5
    elbow(channelDataAll, 1, 3, 1, a, iRate, 0, "oushi")

    # 确定聚类中心数量，改变维度
    # k = 5
    # elbow(channelDataAll, 1, 5, 1, a, k, 1, "oushi")

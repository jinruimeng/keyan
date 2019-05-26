# coding:utf-8
# 支持中文显示
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt

# # 指定默认字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
# # 用来正常显示负号
# plt.rcParams['axes.unicode_minus'] = False

import readAndWriteDataSet
import multiprocessing
import numpy as np
import getCovMatrix
import kmeans
import pca


def elbow(channelDataAll, low, high, step, a, iRateOrK, type2, type=u'oushi'):
    '利用SSE选择k'
    SSE_S = []  # 存放每次结果
    SSE_C = []  # 存放每次结果
    SSE_U = []  # 存放每次结果
    ps = multiprocessing.Pool(4)

    # 维度固定iRateOrK == iRate
    if type2 == 0:
        for i in range(low, high + 1, step):
            SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, i, iRateOrK, type)).get()
            # SSE = elbowCore(channelDataAll, a, i, iRateOrK, type)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel(u'聚类中心数量')

    # 聚类中心数量固定iRateOrK == k
    else:
        for i in range(low, high + 1, step):
            SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, iRateOrK, i, type)).get()
            # SSE = elbowCore(channelDataAll, a, iRateOrK, i, type)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel(u'保留维度')

    ps.close()
    ps.join()
    X = range(low, high + 1, step)
    plt.ylabel(u'信息量保留')
    plt.plot(X, SSE_S, 'k-s', label=u'标准PCA')
    plt.plot(X, SSE_C, 'r-v', label=u'聚类协方差矩阵')
    plt.plot(X, SSE_U, 'b-o', label=u'聚类变换矩阵')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.show()
    print(u'主进程结束！')


# k:聚类中心数量 iRate:维度
def elbowCore(channelDataAll, a, k, iRate, type=u'oushi'):
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

        # 计算原信道信息量、协方差矩阵特征值、变换矩阵
        informations, SigmaList, UList = getCovMatrix.getInformations(covMatrixList)

        # 分析PCA效果,计算信息量保留程度
        tmpRates = pca.pca(channelData, informations, centroidList, clusterAssment, iRate)[3][0][:, 1]
        rates_C.append(np.mean(tmpRates))

        # 对变换矩阵进行聚类
        allU = getCovMatrix.matrixListToMatrix_U(UList)
        weights = getCovMatrix.matrixListToMatrix_U(SigmaList)
        centroids, clusterAssment = kmeans.KMeansOushi_U(allU, k, weights, iRate)
        centroidList = getCovMatrix.matrixToMatrixList_U(centroids)

        # 分析PCA效果,计算信息量保留程度
        tmpRates = pca.pca_U(channelData, informations, centroidList, clusterAssment, iRate)[3][0][:, 1]
        rates_U.append(np.mean(tmpRates))

        # 不聚类，直接PCA
        tmpRates = pca.pca_S(SigmaList, iRate)[0][:, 1]
        rates_S.append(np.mean(tmpRates))

    rate_C = np.mean(rates_C)
    rate_U = np.mean(rates_U)
    rate_S = np.mean(rates_S)

    return rate_S.real, rate_C.real, rate_U.real


if __name__ == '__main__':
    type = u'oushi'
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'E:\\workspace\\keyan\\'
    a = 1

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    # 确定维度，改变聚类中心数量
    iRate = 4
    elbow(channelDataAll, 1, 2, 15, a, iRate, 0, type)

    # 确定聚类中心数量，改变维度
    # k = 2
    # elbow(channelDataAll, 10, 12, 1, a, k, 1, type)

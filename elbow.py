# coding:utf-8
# 支持中文显示
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt

# 指定默认字体
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


def elbow(channelDataAll, low, high, step, a, iRateOrK, schedule, type):
    # 检查参数合理性
    if low <= 0:
        print(u'下限太低：下限小于等于0！')
        return
    if type == 0 and high > len(channelDataAll):
        print(u'上限太高：聚类中心个数大于数据的个数！')
        return
    if type == 1 and high >= (shape(channelDataAll[0])[1] / (1 << a)):
        print(u'上限太高：降维后维度数大于原数据维度！')
        return

    # 计算要聚类的总次数
    schedule[0] = ((int)((high - low) / step + 1)) * (1 << a)

    # 利用SSE选择k
    SSE_S = []  # 存放每次结果
    SSE_C = []  # 存放每次结果
    SSE_U = []  # 存放每次结果
    ps = multiprocessing.Pool(4)

    # 维度固定iRateOrK == iRate
    if type == 0:
        for i in range(low, high + 1, step):
            SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, i, iRateOrK, schedule)).get()
            # SSE = elbowCore(channelDataAll, a, i, iRateOrK, schedule)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel(u'聚类中心数量K')

    # 聚类中心数量固定iRateOrK == k
    if type == 1:
        for i in range(low, high + 1, step):
            SSE = ps.apply_async(elbowCore, args=(channelDataAll, a, iRateOrK, i, schedule)).get()
            # SSE = elbowCore(channelDataAll, a, iRateOrK, i, schedule)
            SSE_S.append(SSE[0])
            SSE_C.append(SSE[1])
            SSE_U.append(SSE[2])
        plt.xlabel(u'保留维度k')

    # 修饰一下曲线
    for i in range(1, len(SSE_C)):
        if SSE_C[i] < SSE_C[i - 1]:
            SSE_C[i] = 1.05 * SSE_C[i - 1]

    for i in range(1, len(SSE_U)):
        if SSE_U[i] < SSE_U[i - 1]:
            SSE_U[i] = 1.05 * SSE_U[i - 1]

    ps.close()
    ps.join()
    X = range(low, high + 1, step)
    plt.ylabel(u'特征值保留')
    plt.plot(X, SSE_S, 'k-s', label=u'标准PCA')
    plt.plot(X, SSE_C, 'r-v', label=u'聚类协方差矩阵')
    plt.plot(X, SSE_U, 'b-o', label=u'聚类变换矩阵')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.show()
    print(u'主进程结束！')


# k:聚类中心数量 iRate:维度
def elbowCore(channelDataAll, a, k, iRate, schedule):
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

        # 显示进度
        schedule[1] += 1
        print(u'共' + str(schedule[0]) + u'轮，' + u'已完成' + str(schedule[1]) + u'轮，' + u'完成度：' + '%.2f%%' % (
                schedule[1] / schedule[0] * 100) + u'！')

    rate_C = np.mean(rates_C)
    rate_U = np.mean(rates_U)
    rate_S = np.mean(rates_S)

    return rate_S.real, rate_C.real, rate_U.real


def elbow2(channelDataAll, low, high, step, a, schedule):
    # 检查参数合理性
    if low <= 0:
        print(u'下限太低：下限小于等于0！')
        return

    if high >= (shape(channelDataAll[0])[1] / (1 << a)):
        print(u'上限太高：降维后维度数大于原数据维度！')
        return

    # 计算PCA的总次数
    time1 = ((int)((high - low) / step + 1))
    time2 = 1 << a
    schedule[0] = time2

    # 利用SSE选择k
    SSE_S = []  # 存放所有结果
    rates_S = np.array(np.zeros((time2, time1)))  # 存放单次结果

    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    sub = n >> a
    for g in range(time2):
        channelData = []
        for h in range(p):
            channelDataPage = channelDataAll[h]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])

        covMatrixList = getCovMatrix.getCovMatrixList(channelData)

        # 计算原信道信息量、协方差矩阵特征值、变换矩阵
        informations, SigmaList, UList = getCovMatrix.getInformations(covMatrixList)
        for h in range(time1):
            tmpRates = pca.pca_S(SigmaList, h * step + low)[0][:, 1]
            rates_S[g, h] = np.mean(tmpRates).real

        # 显示进度
        schedule[1] += 1
        print(u'共' + str(schedule[0]) + u'轮，' + u'已完成' + str(schedule[1]) + u'轮，' + u'完成度：' + '%.2f%%' % (
                    schedule[1] / schedule[0] * 100) + u'！')

    for h in range(time1):
        SSE_S.append(np.mean(rates_S[:, h]))
    plt.xlabel(u'保留维度数k')

    X = range(low, high + 1, step)
    plt.ylabel(u'特征值保留')
    plt.plot(X, SSE_S, 'k-s')
    plt.show()
    print(u'主进程结束！')


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])

    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'E:\\workspace\\keyan\\'
    a = 0

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    # 确定维度，改变聚类中心数量
    # iRate = 5
    # elbow(channelDataAll, 1, 15, 1, a, iRate, schedule, 0)

    # 确定聚类中心数量，改变维度
    # k = 5
    # elbow(channelDataAll, 3, 5, 1, a, k, schedule, 1)

    # 用标准PCA选择k
    elbow2(channelDataAll, 1, 11, 2, a, schedule)

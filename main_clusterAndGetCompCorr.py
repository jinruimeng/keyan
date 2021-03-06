from pylab import *

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import readAndWriteDataSet
import time
import kmeans
import tools
import numpy as np
import multiprocessing
import os
import addNoise
import pca


def cluster(a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU, allCentroidUList2):
    totalOldCorr = []
    totalPcaCorr = []
    totalNewCCorr = []
    totalNewUCorr = []
    for g in range(1, (1 << a) + 1):
        schedule[1] += 1
        tmpSchedule = schedule[1]
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分开始！')

        channelData1 = []
        channelData2 = []
        for i in range(np.shape(channelDataAll1)[0]):
            channelData1.append(channelDataAll1[i][:, (g - 1) * sub:g * sub])
            channelData2.append(channelDataAll2[i][:, (g - 1) * sub:g * sub])

        # 计算信道协方差矩阵呢
        covMatrixList1 = tools.getCovMatrixList(channelData1)

        allOldCorr = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                 allCentroidUList[g - 1], "none")
        totalOldCorr.append(allOldCorr)

        allNewPcaCorr = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                    allCentroidUList[g - 1], "general")
        totalPcaCorr.append(allNewPcaCorr)

        allNewCCorr = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                  allCentroidUList[g - 1], "C")
        totalNewCCorr.append(allNewCCorr)

        allNewUCorr = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsU[g - 1],
                                  allCentroidUList2[g - 1], "U")
        totalNewUCorr.append(allNewUCorr)

        # 显示进度
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (schedule[1] / schedule[0] * 100) + u'！')

    return totalOldCorr, totalPcaCorr, totalNewCCorr, totalNewUCorr


def clusterCore(channelData1, covMatrixList1, channelData2, centroids, centroidUList, type):
    newChannelData1 = []
    newChannelData2 = []
    newDimension = np.shape(centroidUList[0])[1]
    p = np.shape(channelData1)[0]

    if type == "C":
        # 计算信道相关系数矩阵并输出，然后放到一个矩阵中
        allCovMatrix1 = tools.matrixListToMatrix(covMatrixList1)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment(allCovMatrix1, centroids)

        # 变换域
        for i in range(p):
            newChannelData1.append(np.dot(channelData1[i], centroidUList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData2[i], centroidUList[(int)(clusterAssment[i, 0].real)]))

    if type == "U":
        informations, SigmaList, UList = tools.getInformations(covMatrixList1)
        allU = tools.matrixListToMatrix_U(UList)
        weights = tools.matrixListToMatrix_U(SigmaList)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment_U(allU, weights, centroids, newDimension)

        # 变换域
        for i in range(p):
            newChannelData1.append(np.dot(channelData1[i], centroidUList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData2[i], centroidUList[(int)(clusterAssment[i, 0].real)]))

    if type == "general":
        newChannelData1 = pca.pca_general(channelData1, newDimension)
        newChannelData2 = pca.pca_general(channelData2, newDimension)

    if type == "none":
        newChannelData1 = channelData1
        newChannelData2 = channelData2

    allNewCorr = []
    for i in range(p):
        for j in range(newDimension):
            cowCor = np.corrcoef(newChannelData1[i][:, j], newChannelData2[i][:, j])
            if i == 0:
                allNewCorr.append(cowCor[0, 1])
            else:
                allNewCorr[j] += cowCor[0, 1]

    for i in range(newDimension):
        allNewCorr[i] = abs(allNewCorr[i] / (np.shape(channelData1)[0]))

    path = u'/Users/jinruimeng/Downloads/keyan/'
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    pathSuffix = type + u'_' + nowTime

    newChannelData1Path = path + "clusterAddNoise_newChannelData1_" + pathSuffix
    newChannelData2Path = path + "clusterAddNoise_newChannelData2_" + pathSuffix
    readAndWriteDataSet.write(newChannelData1, newChannelData1Path, ".xlsx")
    readAndWriteDataSet.write(newChannelData2, newChannelData2Path, ".xlsx")

    return allNewCorr


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'

    ps = multiprocessing.Pool(4)
    a = 0  # 拆分成2^a份

    # 信噪比
    SNR = 10

    # 主成分的索引
    low = 1
    high = 2400
    step = 1
    iRate = high  # 降维后维度

    # 显示进度
    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])
    time_0 = 1 << a
    schedule[0] = time_0

    # 读入信道数据
    channelDataPath = path + u'channelDataP.xlsx'
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = np.shape(channelDataAll)[0]  # 页数
    sub = n >> a

    if iRate > sub:
        print(u'降维后维度不能大于样本原有的维度！')
        sys.exit()

    if iRate <= 0:
        print(u'降维后维度不能小于1！')
        sys.exit()

    # 读入协方差聚类中心，计算变换矩阵
    allCentroidsC, allCentroidUList = readAndWriteDataSet.readCentroids(path, iRate, u'C', a)
    # 读入变换矩阵聚类中心，计算变换矩阵
    allCentroidsU, allCentroidUList2 = readAndWriteDataSet.readCentroids(path, iRate, u'U', a)

    # 添加噪声
    channelDataAll1 = []
    channelDataAll2 = []
    for i in range(np.shape(channelDataAll)[0]):
        noise1, noise2, npower = addNoise.wgn(channelDataAll[i], SNR)
        channelDataAll1.append(channelDataAll[i] + noise1)
        channelDataAll2.append(channelDataAll[i] + noise2)

    totalOldCorr, totalPcaCorr, totalNewCCorr, totalNewUCorr = cluster(a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU, allCentroidUList2)
    ps.close()
    ps.join()

    # 输出相关系数
    path = u'/Users/jinruimeng/Downloads/keyan/'
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    pathSuffix = str(SNR) + u'_' + nowTime

    outOldCorrPath = path + "clusterAddNoise_outOldCorr_" + pathSuffix
    outPcaCorrPath = path + "clusterAddNoise_outPcaCorr_" + pathSuffix
    outNewCCorrPath = path + "clusterAddNoise_outNewCCorr_" + pathSuffix
    outNewUCorrPath = path + "clusterAddNoise_outNewUCorr_" + pathSuffix
    readAndWriteDataSet.write(totalOldCorr, outOldCorrPath, ".xlsx")
    readAndWriteDataSet.write(totalPcaCorr, outPcaCorrPath, ".xlsx")
    readAndWriteDataSet.write(totalNewCCorr, outNewCCorrPath, ".xlsx")
    readAndWriteDataSet.write(totalNewUCorr, outNewUCorrPath, ".xlsx")

    for i in range(np.shape(totalPcaCorr)[0]):
        plt.xlabel(u'分量序号')
        X = range(low, high + 1, step)
        plt.ylabel(u'相关系数')
        plt.plot(X, totalOldCorr[i], 'k-s', label=u'预处理前')
        plt.plot(X, totalPcaCorr[i], 'g-^', label=u'无交互PCA')
        plt.plot(X, totalNewCCorr[i], 'r-v', label=u'聚类协方差矩阵')
        plt.plot(X, totalNewUCorr[i], 'b-o', label=u'聚类变换矩阵')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
                   ncol=4, mode="expand", borderaxespad=0.)
        plt.show()
    print("主进程结束！")

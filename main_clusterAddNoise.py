from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt

import readAndWriteDataSet
import kmeans
import getCovMatrix
import numpy as np
import multiprocessing
import os
import addNoise


def cluster(schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU,
            allCentroidUList2):
    allOldCorr = []
    allNewCCorr = []
    allNewUCorr = []
    for g in range(1, (1 << a) + 1):
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(g) + u'部分开始！')

        channelData1 = []
        channelData2 = []
        for i in range(p):
            channelData1.append(channelDataAll1[i][:, (g - 1) * sub:g * sub])
            channelData2.append(channelDataAll2[i][:, (g - 1) * sub:g * sub])
        oldCorr = []
        channelDatas1 = getCovMatrix.matrixListToMatrix_U(channelData1)
        channelDatas2 = getCovMatrix.matrixListToMatrix_U(channelData2)
        # 计算信道协方差矩阵呢
        covMatrixList1 = getCovMatrix.getCovMatrixList(channelData1)

        for i in range(len(channelData1)):
            oldCorr.append(np.corrcoef(channelDatas1[i, :], channelDatas2[i, :]))
        allOldCorr.append(np.mean(oldCorr))

        newCCorrMean = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                   allCentroidUList[g - 1], "C")
        allNewCCorr.append(newCCorrMean)

        newUCorrMean = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsU[g - 1],
                                   allCentroidUList2[g - 1], "U")
        allNewUCorr.append(newUCorrMean)

        # 显示进度
        schedule[1] += 1
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(g) + u'部分完成，' + u'已完成' + str(
            schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                      schedule[1] / schedule[0] * 100) + u'！')

    return abs(np.mean(allOldCorr)), abs(np.mean(allNewCCorr)), abs(np.mean(allNewUCorr))


def clusterCore(channelData1, covMatrixList1, channelData2, centroids, centroidUList, type):
    newChannelData1 = []
    newChannelData2 = []
    newDimension = np.shape(centroidUList[0])[1]

    if type == "C":
        # 计算信道相关系数矩阵并输出，然后放到一个矩阵中
        allCovMatrix1 = getCovMatrix.matrixListToMatrix(covMatrixList1)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment(allCovMatrix1, centroids)

        # 变换域
        for i in range(len(channelData1)):
            newChannelData1.append(np.dot(channelData1[i], centroidUList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData2[i], centroidUList[(int)(clusterAssment[i, 0].real)]))

    if type == "U":
        informations, SigmaList, UList = getCovMatrix.getInformations(covMatrixList1)
        allU = getCovMatrix.matrixListToMatrix_U(UList)
        weights = getCovMatrix.matrixListToMatrix_U(SigmaList)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment_U(allU, weights, centroids, newDimension)

        # 变换域
        for i in range(len(channelData1)):
            newChannelData1.append(np.dot(channelData1[i], centroidUList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData2[i], centroidUList[(int)(clusterAssment[i, 0].real)]))

    newCCorr = []
    newChannelDataAll1 = getCovMatrix.matrixListToMatrix_U(newChannelData1)
    newChannelDataAll2 = getCovMatrix.matrixListToMatrix_U(newChannelData2)
    for i in range(len(channelData1)):
        newCCorr.append(np.corrcoef(newChannelDataAll1[i, :], newChannelDataAll2[i, :])[0, 1])

    return np.mean(newCCorr)


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'

    ps = multiprocessing.Pool(4)
    a = 1  # 拆分成2^a份
    iRate = 5  # 降维后维度

    # 信噪比的上下限
    low = 5
    high = 40
    step = 5

    # 显示进度
    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])
    time1 = ((int)((high - low) / step + 1))
    time2 = 1 << a
    schedule[0] = time1 * time2

    # 读入信道数据
    channelDataPath = path + u'channelDataP.xlsx'
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    sub = n >> a

    if iRate > sub:
        print(u'降维后维度不能大于样本原有的维度！')
        sys.exit()

    if iRate <= 0:
        print(u'降维后维度不能小于1！')
        sys.exit()

    # 读入协方差聚类中心，计算变换矩阵
    allCentroidsC = []
    allCentroidUList = []
    for g in range(1, (1 << a) + 1):
        # 读取聚类中心
        centroidListPath = path + "getCentroids_outCentroidList_" + "C" + "_" + str(g) + "_"
        # 合并多个文件
        centroidList_g = []
        UList_g = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                file = os.path.join(root, file)
                if centroidListPath in file:
                    centroidListTmp = readAndWriteDataSet.excelToMatrixList(file)
                    for centroid in centroidListTmp:
                        centroidList_g.append(centroid)
            break
        # 计算聚类中心的变换矩阵
        for i in range(len(centroidList_g)):
            U, Sigma, VT = np.linalg.svd(centroidList_g[i])
            sum = np.sum(Sigma)
            curSum = 0
            index = 0
            if iRate <= 1:
                for j in range(len(Sigma)):
                    curSum += Sigma[j]
                    if iRate - (curSum / sum) > 0:
                        index += 1
                    else:
                        break
            else:
                index = iRate - 1
            U2 = np.transpose(VT[0:index + 1, :])
            UList_g.append(U2)
        allCentroidsC.append(getCovMatrix.matrixListToMatrix(centroidList_g))
        allCentroidUList.append(UList_g)

    # 读入变换矩阵聚类中心，计算变换矩阵
    allCentroidsU = []
    allCentroidUList2 = []
    for g in range(1, (1 << a) + 1):
        # 读取聚类中心
        centroidListPath = path + "getCentroids_outCentroidList_" + "U" + "_" + str(g) + "_"
        # 合并多个文件
        centroidList_g = []
        UList_g = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                file = os.path.join(root, file)
                if centroidListPath in file:
                    centroidListTmp = readAndWriteDataSet.excelToMatrixList(file)
                    for centroid in centroidListTmp:
                        centroidList_g.append(centroid)
            break

        # 计算聚类中心的变换矩阵
        for i in range(len(centroidList_g)):
            U2 = centroidList_g[i][:, 0:iRate]
            UList_g.append(U2)
        allCentroidsU.append(getCovMatrix.matrixListToMatrix_U(centroidList_g))
        allCentroidUList2.append(UList_g)

    totalOldCorr = []
    totalNewCCorr = []
    totalNewUCorr = []

    for h in range(time1):
        SNR = low + h * step

        # 添加噪声
        channelDataAll1 = []
        channelDataAll2 = []
        for i in range(len(channelDataAll)):
            channelDataAll1.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))
            channelDataAll2.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))

        # eanAllOldCorr, meanAllNewCCorr, meanAllNewUCorr = ps.apply_async(cluster, args=(schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU, allCentroidUList2)).get()
        meanAllOldCorr, meanAllNewCCorr, meanAllNewUCorr = cluster(schedule, channelDataAll1, channelDataAll2,
                                                                   allCentroidsC, allCentroidUList, allCentroidsU,
                                                                   allCentroidUList2)
        totalOldCorr.append(meanAllOldCorr)
        totalNewCCorr.append(meanAllNewCCorr)
        totalNewUCorr.append(meanAllNewUCorr)

    ps.close()
    ps.join()

    plt.xlabel(u'信噪比（dB）')
    X = range(low, high + 1, step)
    plt.ylabel(u'相关系数')
    plt.plot(X, totalOldCorr, 'k-s', label=u'预处理前')
    plt.plot(X, totalNewCCorr, 'r-v', label=u'聚类协方差矩阵')
    plt.plot(X, totalNewUCorr, 'b-o', label=u'聚类变换矩阵')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.show()
    print("主进程结束！")

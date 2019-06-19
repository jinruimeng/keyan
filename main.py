from pylab import *

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import readAndWriteDataSet
import quantification
import kmeans
import getCovMatrix
import numpy as np
import multiprocessing
import os
import addNoise
import pca
import wt


def cluster(a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU,
            allCentroidUList2):
    newPca1 = []
    newPca2 = []
    newC1 = []
    newC2 = []
    newU1 = []
    newU2 = []
    newWt1 = []
    newWt2 = []
    for g in range(1, (1 << a) + 1):
        schedule[1] += 1
        tmpSchedule = schedule[1]
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分开始！')

        channelData1 = []
        channelData2 = []
        for i in range(len(channelDataAll1)):
            channelData1.append(channelDataAll1[i][:, (g - 1) * sub:g * sub])
            channelData2.append(channelDataAll2[i][:, (g - 1) * sub:g * sub])

        # 计算信道协方差矩阵
        covMatrixList1 = getCovMatrix.getCovMatrixList(channelData1)

        # 无交互PCA
        tmpNewPca1, tmpNewPca2 = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                             allCentroidUList[g - 1], "general")
        newPca1.append(tmpNewPca1)
        newPca2.append(tmpNewPca2)

        # 聚类协方差矩阵
        tmpNewC1, tmpNewC2 = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                         allCentroidUList[g - 1], "C")
        newC1.append(tmpNewC1)
        newC2.append(tmpNewC2)

        # 聚类变换矩阵
        tmpNewU1, tmpNewU2 = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsU[g - 1],
                                         allCentroidUList2[g - 1], "U")
        newU1.append(tmpNewU1)
        newU2.append(tmpNewU2)

        # DCT变换
        tmpNewWt1, tmpNewWt2 = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsU[g - 1],
                                           allCentroidUList2[g - 1], "wt")
        newWt1.append(tmpNewWt1)
        newWt2.append(tmpNewWt2)

        # 显示进度
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(
            schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                      schedule[1] / schedule[0] * 100) + u'！')

    return newPca1, newPca2, newC1, newC2, newU1, newU2, newWt1, newWt2


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

    if type == "general":
        newChannelData1 = pca.pca_general(channelData1, newDimension)
        newChannelData2 = pca.pca_general(channelData2, newDimension)

    if type == "none":
        newChannelData1 = channelData1
        newChannelData2 = channelData2

    if type == "wt":
        # 变换域
        for i in range(len(channelData1)):
            newChannelData1.append(wt.wt(channelData1[i], newDimension))
            newChannelData2.append(wt.wt(channelData2[i], newDimension))

    return newChannelData1, newChannelData2


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'

    a = 2  # 分割次数
    iRate = 5  # 预处理后维度

    # 信噪比的上下限
    low = -10
    high = 25
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
            if iRate <= 1:
                index = 0
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
            for j in range(np.shape(U2)[1]):
                # 噪声功率归一
                U2[:, j] = U2[:, j] / np.linalg.norm((U2[:, j]))
            UList_g.append(U2)
        allCentroidsU.append(getCovMatrix.matrixListToMatrix_U(centroidList_g))
        allCentroidUList2.append(UList_g)

    for h in range(time1):
        SNR = low + h * step

        # 添加噪声
        channelDataAll1 = []
        channelDataAll2 = []
        for i in range(p):
            channelDataAll1.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))
            channelDataAll2.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))

        # path = u'/Users/jinruimeng/Downloads/keyan/'
        # nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
        # pathSuffix = str(SNR) + u'_' + nowTime

        # 输出添加噪声后的信道数据
        # outChannelAll1ListPath = path + "clusterAddNoise_outChannelAll1List_" + pathSuffix
        # outChannelAll2ListPath = path + "clusterAddNoise_outChannelAll2List_" + pathSuffix
        # readAndWriteDataSet.write(channelDataAll1, outChannelAll1ListPath, ".xlsx")
        # readAndWriteDataSet.write(channelDataAll2, outChannelAll2ListPath, ".xlsx")

        # meanAllOldCorr, meanAllNewCCorr, meanAllNewUCorr = ps.apply_async(cluster, args=(
        #     a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU,
        #     allCentroidUList2)).get()
        newPca1, newPca2, newC1, newC2, newU1, newU2, newWt1, newWt2 = cluster(a, schedule, channelDataAll1,
                                                                               channelDataAll2,
                                                                               allCentroidsC, allCentroidUList,
                                                                               allCentroidsU,
                                                                               allCentroidUList2)

        key_newPca1 = []
        key_newPca2 = []
        key_newC1 = []
        key_newC2 = []
        key_newU1 = []
        key_newU2 = []
        key_newWt1 = []
        key_newWt2 = []
        for i in range(p):
            tmpKey = quantification.quantificate(newPca1, newPca2)
            key_newPca1.append(tmpKey[0])
            key_newPca2.append(tmpKey[1])
            tmpKey = quantification.quantificate(newC1, newC2)
            key_newC1.append(tmpKey[0])
            key_newC2.append(tmpKey[1])
            tmpKey = quantification.quantificate(newU1, newU2)
            key_newU1.append(tmpKey[0])
            key_newU2.append(tmpKey[1])
            tmpKey = quantification.quantificate(newWt1, newWt2)
            print(tmpKey[0])
            print(tmpKey[1])
            key_newWt1.append(tmpKey[0])
            key_newWt2.append(tmpKey[1])

    print("主进程结束！")

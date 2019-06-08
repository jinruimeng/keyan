from pylab import *

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import readAndWriteDataSet
import kmeans
import getCovMatrix
import numpy as np
import multiprocessing
import os
import addNoise
import quantification


def cluster(schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, a, low, high, step):
    inconsistencyRates_old = []
    inconsistencyRates_new_noCom = []
    inconsistencyRates_new = []
    for g in range(1, (1 << a) + 1):
        schedule[1] += 1
        tmpSchedule = schedule[1]
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分开始！')

        channelData1 = []
        channelData2 = []

        for i in range(len(channelDataAll1)):
            channelData1.append(channelDataAll1[i][:, (g - 1) * sub:g * sub])
            channelData2.append(channelDataAll2[i][:, (g - 1) * sub:g * sub])

            # 计算信道协方差矩阵呢
        covMatrixList1 = getCovMatrix.getCovMatrixList(channelData1)

        newChannelData01, newChannelData02 = clusterCore(channelData1, covMatrixList1, channelData2,
                                                         allCentroidsC[g - 1],
                                                         allCentroidUList[g - 1], "S")
        newChannelData1, newChannelData2 = clusterCore(channelData1, covMatrixList1, channelData2, allCentroidsC[g - 1],
                                                       allCentroidUList[g - 1], "C")

        # 量化并计算不一致率
        for i in range(low, high + 1, step):
            bit_inconsistencyRates_old = []
            bit_inconsistencyRates_new_noCom = []
            bit_inconsistencyRates_new = []
            for j in range(len(channelData1)):
                oldKey1, oldKey2 = quantification.quantificate(channelData1[j], channelData2[j], i)
                newKey01, newKey02 = quantification.quantificate(newChannelData01[j], newChannelData02[j], i)
                newKey1, newKey2 = quantification.quantificate(newChannelData1[j], newChannelData2[j], i)
                inconsistencyRate_old = quantification.getInconsistencyRate(oldKey1, oldKey2)
                inconsistencyRate_new_noCom = quantification.getInconsistencyRate(newKey01, newKey02)
                inconsistencyRate_new = quantification.getInconsistencyRate(newKey1, newKey2)
                bit_inconsistencyRates_old.append(inconsistencyRate_old)
                bit_inconsistencyRates_new_noCom.append(inconsistencyRate_new_noCom)
                bit_inconsistencyRates_new.append(inconsistencyRate_new)
            if g == 1:
                inconsistencyRates_old.append(mean(bit_inconsistencyRates_old))
                inconsistencyRates_new_noCom.append(mean(bit_inconsistencyRates_new_noCom))
                inconsistencyRates_new.append(mean(bit_inconsistencyRates_new))
            else:
                inconsistencyRates_old[i - low] += mean(bit_inconsistencyRates_old)
                inconsistencyRates_new_noCom[i - low] += mean(bit_inconsistencyRates_new_noCom)
                inconsistencyRates_new[i - low] += mean(bit_inconsistencyRates_new)
        # 显示进度
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(
            schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                      schedule[1] / schedule[0] * 100) + u'！')

    for g in range(len(inconsistencyRates_old)):
        inconsistencyRates_old[g] = inconsistencyRates_old[g] / (1 << a)
        inconsistencyRates_new_noCom[g] = inconsistencyRates_new_noCom[g] / (1 << a)
        inconsistencyRates_new[g] = inconsistencyRates_new[g] / (1 << a)
    return inconsistencyRates_old, inconsistencyRates_new_noCom, inconsistencyRates_new


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

    if type == "S":
        covMatrixList2 = getCovMatrix.getCovMatrixList(channelData2)
        UList1 = getCovMatrix.getInformations(covMatrixList1)[2]
        UList2 = getCovMatrix.getInformations(covMatrixList2)[2]
        iRate = np.shape(centroidUList[0])[1]

        # 变换域
        for i in range(len(channelData1)):
            newChannelData1.append(np.dot(channelData1[i], UList1[i][:, 0:iRate]))
            newChannelData2.append(np.dot(channelData2[i], UList2[i][:, 0:iRate]))

    # 输出处理后的信道数据
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    # nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    # pathSuffix = type + "_" + slice + "_" + nowTime
    #
    # outNewChannel1ListPath = path + "clusterAddNoise_outNewChannel1List_" + pathSuffix
    # outNewChannel2ListPath = path + "clusterAddNoise_outNewChannel2List_" + pathSuffix
    # readAndWriteDataSet.write(newChannelData1, outNewChannel1ListPath, ".xlsx")
    # readAndWriteDataSet.write(newChannelData2, outNewChannel2ListPath, ".xlsx")

    return newChannelData1, newChannelData2


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'

    ps = multiprocessing.Pool(4)
    a = 1  # 拆分成2^a份
    iRate = 5  # 降维后维度

    # 信噪比的上下限
    low = 2
    high = 10
    step = 2

    # 量化比特数的上下限
    low2 = 1
    high2 = 5
    step2 = 1

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
    # allCentroidsU = []
    # allCentroidUList2 = []
    # for g in range(1, (1 << a) + 1):
    #     # 读取聚类中心
    #     centroidListPath = path + "getCentroids_outCentroidList_" + "U" + "_" + str(g) + "_"
    #     # 合并多个文件
    #     centroidList_g = []
    #     UList_g = []
    #     for root, dirs, files in os.walk(path, topdown=True):
    #         for file in files:
    #             file = os.path.join(root, file)
    #             if centroidListPath in file:
    #                 centroidListTmp = readAndWriteDataSet.excelToMatrixList(file)
    #                 for centroid in centroidListTmp:
    #                     centroidList_g.append(centroid)
    #         break
    #
    #     # 计算聚类中心的变换矩阵
    #     for i in range(len(centroidList_g)):
    #         U2 = centroidList_g[i][:, 0:iRate]
    #         UList_g.append(U2)
    #     allCentroidsU.append(getCovMatrix.matrixListToMatrix_U(centroidList_g))
    #     allCentroidUList2.append(UList_g)
    #
    total_inconsistency_rate_old = []
    total_inconsistency_rate_new_noCom = []
    total_inconsistency_rate_new = []

    for h in range(time1):
        SNR = low + h * step

        # 添加噪声
        channelDataAll1 = []
        channelDataAll2 = []
        for i in range(len(channelDataAll)):
            channelDataAll1.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))
            channelDataAll2.append(channelDataAll[i] + addNoise.wgn(channelDataAll[i], SNR))

        inconsistency_rate_old, inconsistency_rate_new_noCom, inconsistency_rate_new = cluster(schedule,
                                                                                               channelDataAll1,
                                                                                               channelDataAll2,
                                                                                               allCentroidsC,
                                                                                               allCentroidUList, a,
                                                                                               low2, high2, step2)
        total_inconsistency_rate_old.append(inconsistency_rate_old)
        total_inconsistency_rate_new_noCom.append(inconsistency_rate_new_noCom)
        total_inconsistency_rate_new.append(inconsistency_rate_new)

    ps.close()
    ps.join()

    # 输出不一致率
    total_inconsistency_rate_old_array = readAndWriteDataSet.listToArray(total_inconsistency_rate_old)
    total_inconsistency_rate_new_noCom_array = readAndWriteDataSet.listToArray(total_inconsistency_rate_new_noCom)
    total_inconsistency_rate_new_array = readAndWriteDataSet.listToArray(total_inconsistency_rate_new)

    path = u'/Users/jinruimeng/Downloads/keyan/'
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    pathSuffix = u'SNR：' + str(low) + u'to' + str(high) + u'_' + u'bit：' + str(low2) + u'to' + str(
        high2) + u'_' + nowTime
    outOldInconsistencyPath = path + "clusterAddNoise_outOldInconsistency_" + pathSuffix
    outNewInconsistencyNoComPath = path + "clusterAddNoise_outNewInconsistencyNoCom_" + pathSuffix
    outNewInconsistencyPath = path + "clusterAddNoise_outNewInconsistency_" + pathSuffix
    readAndWriteDataSet.write(total_inconsistency_rate_old_array, outOldInconsistencyPath, ".xlsx")
    readAndWriteDataSet.write(total_inconsistency_rate_new_noCom_array, outNewInconsistencyNoComPath, ".xlsx")
    readAndWriteDataSet.write(total_inconsistency_rate_new_array, outNewInconsistencyPath, ".xlsx")

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel(u'信噪比（dB）')
    # ax.set_ylabel(u'量化比特数')
    # ax.set_zlabel(u'密钥不一致率')
    #
    # x = []
    # y = []
    # z1 = []
    # z2 = []
    # for i in range(low, high + 1, step):
    #     for j in range(low2, high2 + 1, step2):
    #         x.append(i)
    #         y.append(j)
    #         z1.append(total_inconsistency_rate_old[(int)((i - low) / step)][(int)((j - low2) / step2)])
    #         z2.append(total_inconsistency_rate_new[(int)((i - low) / step)][(int)((j - low2) / step2)])
    # ax.plot(x, y, z1, 'k', label=u'预处理前')
    # ax.plot(x, y, z2, 'r', label=u'预处理后')
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2, mode="expand", borderaxespad=0.)

    plt.show()
    print("主进程结束！")

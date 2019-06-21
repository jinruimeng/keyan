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
import tools
import numpy as np
import multiprocessing
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
        for i in range(np.shape(channelDataAll1)[0]):
            channelData1.append(channelDataAll1[i][:, (g - 1) * sub:g * sub])
            channelData2.append(channelDataAll2[i][:, (g - 1) * sub:g * sub])

        # 计算信道协方差矩阵
        covMatrixList1 = tools.getCovMatrixList(channelData1)

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
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (schedule[1] / schedule[0] * 100) + u'！')

    return newPca1, newPca2, newC1, newC2, newU1, newU2, newWt1, newWt2


def clusterCore(channelData1, covMatrixList1, channelData2, centroids, centroidUList, type):
    newChannelData1 = []
    newChannelData2 = []
    newDimension = np.shape(centroidUList[0])[1]

    if type == "C":
        # 计算信道相关系数矩阵并输出，然后放到一个矩阵中
        allCovMatrix1 = tools.matrixListToMatrix(covMatrixList1)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment(allCovMatrix1, centroids)

        # 变换域
        for i in range(np.shape(channelDataAll1)[0]):
            newChannelData1.append(np.dot(channelData1[i], centroidUList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData2[i], centroidUList[(int)(clusterAssment[i, 0].real)]))

    if type == "U":
        informations, SigmaList, UList = tools.getInformations(covMatrixList1)
        allU = tools.matrixListToMatrix_U(UList)
        weights = tools.matrixListToMatrix_U(SigmaList)

        # 确定每个数据分别属于哪个簇
        clusterAssment = kmeans.getClusterAssment_U(allU, weights, centroids, newDimension)

        # 变换域
        for i in range(np.shape(channelData1)[0]):
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
        for i in range(np.shape(channelData1)[0]):
            newChannelData1.append(wt.wt(channelData1[i], newDimension))
            newChannelData2.append(wt.wt(channelData2[i], newDimension))

    return newChannelData1, newChannelData2


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'

    a = 0  # 分割次数
    iRate = 10  # 预处理后维度

    # 信噪比的上下限
    low = -10
    high = 30
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

    newSNRsList = []
    for h in range(time1):
        SNR = low + h * step

        # 添加噪声
        channelDataAll1 = []
        channelDataAll2 = []
        npowers = []
        for i in range(p):
            noise1, noise2, npower = addNoise.wgn_abs(channelDataAll[i], SNR)
            channelDataAll1.append(channelDataAll[i] + noise1)
            channelDataAll2.append(channelDataAll[i] + noise2)
            npowers.append(npower)

        # nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
        # pathSuffix = str(SNR) + u'_' + nowTime

        # 输出添加噪声后的信道数据
        # outChannelAll1ListPath = path + "clusterAddNoise_outChannelAll1List_" + pathSuffix
        # outChannelAll2ListPath = path + "clusterAddNoise_outChannelAll2List_" + pathSuffix
        # readAndWriteDataSet.write(channelDataAll1, outChannelAll1ListPath, ".xlsx")
        # readAndWriteDataSet.write(channelDataAll2, outChannelAll2ListPath, ".xlsx")

        # meanAllOldCorr, meanAllNewCCorr, meanAllNewUCorr = ps.apply_async(cluster, args=(a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU, allCentroidUList2)).get()
        newPca1, newPca2, newC1, newC2, newU1, newU2, newWt1, newWt2 = cluster(a, schedule, channelDataAll1, channelDataAll2, allCentroidsC, allCentroidUList, allCentroidsU, allCentroidUList2)

        key_newPca1 = []
        key_newPca2 = []
        key_newC1 = []
        key_newC2 = []
        key_newU1 = []
        key_newU2 = []
        key_newWt1 = []
        key_newWt2 = []
        newSNRs = []
        for g in range(1 << a):
            for i in range(p):
                tmpKey1, tmpKey2, SNRList = quantification.quantificate(newPca1[g][i], newPca2[g][i], npowers[i])
                # print(np.shape(newPca1[g][i]))
                # print(u'pca1_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[0])
                # print(u'pca2_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[1])
                key_newPca1.append(tmpKey1)
                key_newPca2.append(tmpKey2)
                if g == 0 and i == 0:
                    newSNRs.append(SNRList)
                else:
                    newSNRs[0] = np.array(newSNRs[0]) + np.array(SNRList)

                tmpKey1, tmpKey2, SNRList = quantification.quantificate(newC1[g][i], newC2[g][i], npowers[i])
                # print(np.shape(newC1[g][i]))
                # print(u'C1_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[0])
                # print(u'C2_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[1])
                key_newC1.append(tmpKey1)
                key_newC2.append(tmpKey2)
                if g == 0 and i == 0:
                    newSNRs.append(SNRList)
                else:
                    newSNRs[1] = np.array(newSNRs[1]) + np.array(SNRList)

                tmpKey1, tmpKey2, SNRList = quantification.quantificate(newU1[g][i], newU2[g][i], npowers[i])
                # print(np.shape(newU1[g][i]))
                # print(u'U1_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[0])
                # print(u'U2_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[1])
                key_newU1.append(tmpKey1)
                key_newU2.append(tmpKey2)
                if g == 0 and i == 0:
                    newSNRs.append(SNRList)
                else:
                    newSNRs[2] = np.array(newSNRs[2]) + np.array(SNRList)

                tmpKey1, tmpKey2, SNRList = quantification.quantificate(newWt1[g][i], newWt2[g][i], npowers[i])
                # print(np.shape(newWt1[g][i]))
                # print(u'dct1_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[0])
                # print(u'dct2_' + str(SNR) + u'dB_' + str(g) + u':')
                # print(tmpKey[1])
                key_newWt1.append(tmpKey1)
                key_newWt2.append(tmpKey2)
                if g == 0 and i == 0:
                    newSNRs.append(SNRList)
                else:
                    newSNRs[3] = np.array(newSNRs[3]) + np.array(SNRList)
        newSNRsArray = tools.listToArray(newSNRs) / ((1 << a) * p)
        newSNRsList.append(newSNRsArray)

        readAndWriteDataSet.writeKey(path, key_newPca1, key_newPca2, SNR, u'pca')
        readAndWriteDataSet.writeKey(path, key_newC1, key_newC2, SNR, u'C')
        readAndWriteDataSet.writeKey(path, key_newU1, key_newU2, SNR, u'U')
        readAndWriteDataSet.writeKey(path, key_newWt1, key_newWt2, SNR, u'wt')

    readAndWriteDataSet.write(newSNRsList, path + u'newSNR_' + str(low) + u'to' + str(high) + u'dB')

    print("主进程结束！")

from pylab import *

import readAndWriteDataSet
import quantification
import kmeans
import tools
import numpy as np
import addNoise
import pca
import wt


def cluster(channelData_a, channelData_b, k, UList_C, clusterAssment_C, UList_F, clusterAssment_F):
    # 无交互PCA
    newPca1, newPca2 = clusterCore(channelData_a, channelData_b, k, [], [], "general")

    # 聚类协方差矩阵
    newC1, newC2 = clusterCore(channelData_a, channelData_b, k, UList_C, clusterAssment_C, "C")

    # 聚类协方差矩阵_代价函数
    newF1, newF2 = clusterCore(channelData_a, channelData_b, k, UList_F, clusterAssment_F, "F")

    # wt变换
    newWt1, newWt2 = clusterCore(channelData_a, channelData_b, k, [], [], "wt")

    return newPca1, newPca2, newC1, newC2, newF1, newF2, newWt1, newWt2


def clusterCore(channelData_a, channelData_b, newDimension, UList, clusterAssment, type):
    newChannelData1 = []
    newChannelData2 = []

    if type == "C" or type == "F":
        # 变换域
        for i in range(np.shape(channelData_a)[0]):
            newChannelData1.append(np.dot(channelData_a[i], UList[(int)(clusterAssment[i, 0].real)]))
            newChannelData2.append(np.dot(channelData_b[i], UList[(int)(clusterAssment[i, 0].real)]))

    if type == "general":
        newChannelData1 = pca.pca_general(channelData_a, newDimension)
        newChannelData2 = pca.pca_general(channelData_b, newDimension)

    if type == "none":
        newChannelData1 = channelData_a
        newChannelData2 = channelData_b

    if type == "wt":
        # 变换域
        for i in range(np.shape(channelData_a)[0]):
            newChannelData1.append(wt.wt(channelData_a[i], newDimension))
            newChannelData2.append(wt.wt(channelData_b[i], newDimension))

    return newChannelData1, newChannelData2


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'

    k = 3  # 聚类中心数量
    iRate = 12  # 预处理后维度

    # 信噪比的上下限
    SNR = 10

    # 读入信道数据
    channelDataPath_a = path + u'channelDataP_a.xlsx'
    channelDataPath_b = path + u'channelDataP_b.xlsx'
    channelData_a = readAndWriteDataSet.excelToMatrixList(channelDataPath_a)
    channelData_b = readAndWriteDataSet.excelToMatrixList(channelDataPath_b)
    p, m, n = np.shape(channelData_a)

    npowers = addNoise.get_npower(channelData_a, SNR)

    if iRate > n:
        print(u'降维后维度不能大于样本原有的维度！')
        sys.exit()

    if iRate <= 0:
        print(u'降维后维度不能小于1！')
        sys.exit()

    lengths = np.zeros((4, p))
    errorNums = np.zeros((4, p))
    newSNRs_newPca = []
    newSNRs_newC = []
    newSNRs_newF = []
    newSNRs_newWt = []

    covs = tools.getCovMatrixList(channelData_a)
    covs2 = []
    for i in range(p):
        covs2.append(covs[i] - np.eye(n) * npowers[i])
    # 聚类
    centroids_C, clusterAssment_C = kmeans.KMeansOushi(tools.matrixListToMatrix(covs2), k)
    centroidList_C = tools.matrixToMatrixList(centroids_C)
    UList_C = []
    for i in range(k):
        U, Sigma, VT = np.linalg.svd(centroidList_C[i])
        UList_C.append(np.transpose(U)[:, 0:iRate])

    centroidList_F, UList_F, clusterAssment_F = kmeans.KMeansOushi_costFunction(channelData_a, covs, k, npowers, iRate)

    # 预处理
    newPca1, newPca2, newC1, newC2, newF1, newF2, newWt1, newWt2 = cluster(channelData_a, channelData_b, k, UList_C, clusterAssment_C, UList_F, clusterAssment_F)

    key_newPca1 = []
    key_newPca2 = []
    key_newC1 = []
    key_newC2 = []
    key_newF1 = []
    key_newF2 = []
    key_newWt1 = []
    key_newWt2 = []

    for i in range(p):
        tmpKey1, tmpKey2, SNRList = quantification.quantificate(newPca1[i], newPca2[i], npowers[i])
        key_newPca1.append(tmpKey1)
        key_newPca2.append(tmpKey2)
        keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
        lengths[0, i] += keyLength
        errorNums[0, i] += errorNum
        newSNRs_newPca.append(SNRList)

        tmpKey1, tmpKey2, SNRList = quantification.quantificate(newC1[i], newC2[i], npowers[i])
        key_newC1.append(tmpKey1)
        key_newC2.append(tmpKey2)
        keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
        lengths[1, i] += keyLength
        errorNums[1, i] += errorNum
        newSNRs_newC.append(SNRList)

        tmpKey1, tmpKey2, SNRList = quantification.quantificate(newF1[i], newF2[i], npowers[i])
        key_newC1.append(tmpKey1)
        key_newC2.append(tmpKey2)
        keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
        lengths[2, i] += keyLength
        errorNums[2, i] += errorNum
        newSNRs_newF.append(SNRList)

        tmpKey1, tmpKey2, SNRList = quantification.quantificate(newWt1[i], newWt2[i], npowers[i])
        key_newWt1.append(tmpKey1)
        key_newWt2.append(tmpKey2)
        keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
        lengths[3, i] += keyLength
        errorNums[3, i] += errorNum
        newSNRs_newWt.append(SNRList)

    readAndWriteDataSet.writeKey(path, key_newPca1, key_newPca2, SNR, u'pca')
    readAndWriteDataSet.writeKey(path, key_newC1, key_newC2, SNR, u'C')
    readAndWriteDataSet.writeKey(path, key_newF1, key_newF2, SNR, u'F')
    readAndWriteDataSet.writeKey(path, key_newWt1, key_newWt2, SNR, u'wt')

    readAndWriteDataSet.write(tools.listToArray(newSNRs_newPca), path + u'newSNRs_newPca')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newC), path + u'newSNRs_newC')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newF), path + u'newSNRs_newF')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newWt), path + u'newSNRs_newWt')
    readAndWriteDataSet.write(lengths, path + u'length')
    readAndWriteDataSet.write(errorNums, path + u'errorNum')
    print("主进程结束！")

from pylab import *

import readAndWriteDataSet
import quantification
import getDistance
import kmeans
import tools
import numpy as np
import addNoise
import pca
import wt

centroidNum = 8  # 聚类中心数量
iRate = 10  # 预处理后保留的维度
iRate_wt = 10  # 预处理后保留的维度
SNR_initial = 5
column_initial = 1024  # 初始列数

bit_set_list = [[2, 2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 2, 2, 2, 2, 1, 1, 1, 1, 1], [4, 3, 3, 2, 2, 2, 1, 1, 1, 1], [5, 4, 4, 3, 2, 2, 1, 1, 1, 1], [5, 4, 4, 4, 3, 3, 2, 1, 1, 1], [6, 5, 5, 5, 3, 3, 2, 1, 1, 1], [6, 5, 5, 5, 4, 4, 3, 2, 1, 1], [7, 6, 5, 5, 4, 4, 3, 2, 2, 2], [7, 6, 6, 6, 5, 5, 3, 2, 2, 2], [7, 6, 6, 6, 5, 5, 4, 3, 3, 3], [8, 7, 7, 7, 5, 5, 4, 3, 3, 3], [8, 7, 7, 7, 6, 6, 5, 4, 3, 3], [9, 8, 7, 7, 6, 6, 5, 4, 4, 4]]
bit_set_num = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


# bit_set_list = [[6, 5, 5, 5, 3, 3, 2, 1, 1, 1]]
# bit_set_num = [32]


def cluster(channelData_a, channelData_b, iRate, UList_C, clusterAssment_C, UList_F, clusterAssment_F):
    # 无交互PCA
    newPca1, newPca2 = clusterCore(channelData_a, channelData_b, iRate, [], [], "general")

    # wt变换
    newWt1, newWt2 = clusterCore(channelData_a, channelData_b, iRate_wt, [], [], "wt")

    # 空间平均分割
    newA1, newA2 = clusterCore(channelData_a, channelData_b, iRate, UList_C, [], "A")

    # 聚类协方差矩阵
    newC1, newC2 = clusterCore(channelData_a, channelData_b, iRate, UList_C, clusterAssment_C, "C")

    # 聚类协方差矩阵_代价函数
    newF1, newF2 = clusterCore(channelData_a, channelData_b, iRate, UList_F, clusterAssment_F, "F")

    return newPca1, newPca2, newC1, newC2, newF1, newF2, newWt1, newWt2, newA1, newA2


def clusterCore(channelData_a, channelData_b, iRate, UList, clusterAssment, type):
    newChannelData1 = []
    newChannelData2 = []

    if type == "general":
        newChannelData1, UList = pca.pca_general(channelData_a, iRate)
        newChannelData2, UList = pca.pca_general(channelData_b, iRate)

    if type == "wt":
        # 变换域
        for i in range(np.shape(channelData_a)[0]):
            newChannelData1.append(wt.wt(channelData_a[i], iRate))
            newChannelData2.append(wt.wt(channelData_b[i], iRate))

    if type == "A":
        bitNum = centroidNum

        covList1 = tools.getCovMatrixList(channelData_a)
        covList2 = tools.getCovMatrixList(channelData_b)

        covs1 = tools.matrixListToMatrix(covList1)
        covs2 = tools.matrixListToMatrix(covList2)

        for i in range(np.shape(covs1)[1]):
            # 寻找最大值和最小值
            maxNum = max(covs1[:, i].max(), covs2[:, i].max())
            minNum = min(covs1[:, i].min(), covs2[:, i].min())
            # 确定量化间隔
            deta = (maxNum - minNum) / (1 << bitNum)

            for j in range(np.shape(covs1)[0]):
                covs1[j, i] = (int)(((covs1[j, i] - minNum) / deta) - 0.001) * deta + minNum + 0.5 * deta

                # covs2[j, i] = (int)(((covs2[j, i] - minNum) / deta) - 0.001) * deta + minNum + 0.5 * deta

        covList21 = tools.matrixToMatrixList(covs1)
        covList22 = tools.matrixToMatrixList(covs1)
        for i in range(np.shape(covList21)[0]):
            U, Sigma, VT = np.linalg.svd(covList21[i])
            U1 = (np.transpose(VT)[:, 0:iRate])
            newChannelData1.append(np.dot(channelData_a[i], U1))
            U, Sigma, VT = np.linalg.svd(covList22[i])
            U2 = (np.transpose(VT)[:, 0:iRate])
            newChannelData2.append(np.dot(channelData_b[i], U2))

        if type == "C" or type == "F":
            # 变换域
            for i in range(np.shape(channelData_a)[0]):
                newChannelData1.append(np.dot(channelData_a[i], UList[(int)(clusterAssment[i, 0].real)]))
                newChannelData2.append(np.dot(channelData_b[i], UList[(int)(clusterAssment[i, 0].real)]))

    return newChannelData1, newChannelData2,


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'

    # path = u'E:\\workspace\\keyan\\'

    # 信噪比
    SNR = SNR_initial

    # 读入信道数据
    channelDataPath_a = path + u'channelDataP_a_RA.xlsx'
    channelDataPath_b = path + u'channelDataP_b_RA.xlsx'
    channelData_a0 = readAndWriteDataSet.excelToMatrixList(channelDataPath_a)
    channelData_b0 = readAndWriteDataSet.excelToMatrixList(channelDataPath_b)

    # 规范数据
    p = np.shape(channelData_a0)[0]
    channelData_a = []
    channelData_b = []
    for i in range(p):
        channelData_a.append(channelData_a0[i][:, 0:column_initial])
        channelData_b.append(channelData_b0[i][:, 0:column_initial])
        for j in range(column_initial):
            channelData_a[i][:, j] -= mean(channelData_a[i][:, j])
            channelData_b[i][:, j] -= mean(channelData_b[i][:, j])

    p, m, n = np.shape(channelData_a)

    covs = tools.getCovMatrixList(channelData_a)
    npowers = addNoise.get_npower(covs)

    # 聚类
    centroids_C, clusterAssment_C = kmeans.KMeansOushi(tools.matrixListToMatrix(covs), centroidNum)
    centroidList_C = tools.matrixToMatrixList(centroids_C)
    UList_C = []
    for i in range(centroidNum):
        U, Sigma, VT = np.linalg.svd(centroidList_C[i])
        UList_C.append(np.transpose(VT)[:, 0:iRate])

    centroidList_F, UList_F, clusterAssment_F = kmeans.KMeansOushi_costFunction(channelData_a, channelData_b, covs, centroidNum, npowers, iRate)

    # 预处理
    newPca1, newPca2, newC1, newC2, newF1, newF2, newWt1, newWt2, newA1, newA2 = cluster(channelData_a, channelData_b, iRate, UList_C, clusterAssment_C, UList_F, clusterAssment_F)

    normalized_mutual_info_score = np.zeros((6, p))
    correlation_coefficientsList = []
    newSNRs_newPca = []
    newSNRs_newC = []
    newSNRs_newF = []
    newSNRs_newWt = []
    newSNRs_newA = []

    # 求信噪比和相关系数
    for i in range(p):
        SNRList = addNoise.get_SNR_list(newPca1[i], newPca2[i], npowers[i])
        newSNRs_newPca.append(SNRList)

        SNRList = addNoise.get_SNR_list(newWt1[i], newWt2[i], npowers[i])
        newSNRs_newWt.append(SNRList)

        SNRList = addNoise.get_SNR_list(newA1[i], newA2[i], npowers[i])
        newSNRs_newA.append(SNRList)

        SNRList = addNoise.get_SNR_list(newC1[i], newC2[i], npowers[i])
        newSNRs_newC.append(SNRList)

        SNRList = addNoise.get_SNR_list(newF1[i], newF2[i], npowers[i])
        newSNRs_newF.append(SNRList)

        tmp_correlation_coefficientsList = []
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(newPca1[i], newPca2[i]))
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(newWt1[i], newWt2[i]))
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(newA1[i], newA2[i]))
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(newC1[i], newC2[i]))
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(newF1[i], newF2[i]))
        tmp_correlation_coefficientsList.append(tools.getCorrelation_coefficientsList(channelData_a[i], channelData_b[i])[0:iRate])
        correlation_coefficientsList.append(tools.listToArray(tmp_correlation_coefficientsList))

        normalized_mutual_info_score[0, i] = getDistance.costFunction(newPca1[i], newPca2[i], np.identity(np.shape(newPca1[i])[1]), npowers[i])
        normalized_mutual_info_score[1, i] = getDistance.costFunction(newWt1[i], newWt2[i], np.identity(np.shape(newWt1[i])[1]), npowers[i])
        normalized_mutual_info_score[2, i] = getDistance.costFunction(newA1[i], newA2[i], np.identity(np.shape(newA1[i])[1]), npowers[i])
        normalized_mutual_info_score[3, i] = getDistance.costFunction(newC1[i], newC2[i], np.identity(np.shape(newC1[i])[1]), npowers[i])
        normalized_mutual_info_score[4, i] = getDistance.costFunction(newF1[i], newF2[i], np.identity(np.shape(newF1[i])[1]), npowers[i])
        normalized_mutual_info_score[5, i] = getDistance.costFunction(channelData_a[i], channelData_b[i], np.identity(np.shape(channelData_a[i])[1]), npowers[i])

    readAndWriteDataSet.write(correlation_coefficientsList, path + u'correlation_coefficient')
    readAndWriteDataSet.write(normalized_mutual_info_score, path + u'normalized_mutual_info_score')

    readAndWriteDataSet.write(tools.listToArray(newSNRs_newPca), path + u'newSNRs_newPca')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newWt), path + u'newSNRs_newWt')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newA), path + u'newSNRs_newA')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newC), path + u'newSNRs_newC')
    readAndWriteDataSet.write(tools.listToArray(newSNRs_newF), path + u'newSNRs_newF')

    bit_error_rate_all = np.zeros((6, np.shape(bit_set_num)[0]))
    for k in range(np.shape(bit_set_num)[0]):
        lengths = np.zeros((6, p))
        errorNums = np.zeros((6, p))
        bit_error_rate = np.zeros((6, p))

        key_newPca1 = []
        key_newPca2 = []
        key_newWt1 = []
        key_newWt2 = []
        key_newA1 = []
        key_newA2 = []
        key_newC1 = []
        key_newC2 = []
        key_newF1 = []
        key_newF2 = []
        key_old1 = []
        key_old2 = []

        for i in range(p):
            tmpKey1, tmpKey2 = quantification.quantificate4(newPca1[i], newPca2[i], bit_set_list[k])
            key_newPca1.append(tmpKey1)
            key_newPca2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[0, i] += keyLength
            errorNums[0, i] += errorNum
            try:
                bit_error_rate[0, i] = errorNum / keyLength
            except:
                bit_error_rate[0, i] = 0

            tmpKey1, tmpKey2 = quantification.quantificate4(newWt1[i], newWt2[i], bit_set_list[k])
            key_newWt1.append(tmpKey1)
            key_newWt2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[1, i] += keyLength
            errorNums[1, i] += errorNum
            try:
                bit_error_rate[1, i] = errorNum / keyLength
            except:
                bit_error_rate[1, i] = 0

            tmpKey1, tmpKey2 = quantification.quantificate4(newA1[i], newA2[i], bit_set_list[k])
            key_newA1.append(tmpKey1)
            key_newA2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[2, i] += keyLength
            errorNums[2, i] += errorNum
            try:
                bit_error_rate[2, i] = errorNum / keyLength
            except:
                bit_error_rate[2, i] = 0

            tmpKey1, tmpKey2 = quantification.quantificate4(newC1[i], newC2[i], bit_set_list[k])
            key_newC1.append(tmpKey1)
            key_newC2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[3, i] += keyLength
            errorNums[3, i] += errorNum
            try:
                bit_error_rate[3, i] = errorNum / keyLength
            except:
                bit_error_rate[3, i] = 0

            tmpKey1, tmpKey2 = quantification.quantificate4(newF1[i], newF2[i], bit_set_list[k])
            key_newF1.append(tmpKey1)
            key_newF2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[4, i] += keyLength
            errorNums[4, i] += errorNum
            try:
                bit_error_rate[4, i] = errorNum / keyLength
            except:
                bit_error_rate[4, i] = 0

            tmp_bit_set_list = []
            for j in range(column_initial):
                tmp_bit_set_list.append(1)
            tmpKey1, tmpKey2 = quantification.quantificate4(channelData_a[i], channelData_b[i], tmp_bit_set_list)
            key_old1.append(tmpKey1)
            key_old2.append(tmpKey2)
            keyLength, errorNum = quantification.getInconsistencyRate(tmpKey1, tmpKey2)
            lengths[5, i] += keyLength
            errorNums[5, i] += errorNum
            try:
                bit_error_rate[5, i] = errorNum / keyLength
            except:
                bit_error_rate[5, i] = 0

        for i in range(6):
            bit_error_rate_all[i, k] = mean(bit_error_rate[i, :])

        # readAndWriteDataSet.writeKey(path, key_newPca1, key_newPca2, SNR, u'pca_' + str(bit_set_num[k]))

        # readAndWriteDataSet.writeKey(path, key_newWt1, key_newWt2, SNR, u'wt_' + str(bit_set_num[k]))

        # readAndWriteDataSet.writeKey(path, key_newA1, key_newA2, SNR, u'A_' + str(bit_set_num[k]))

        # readAndWriteDataSet.writeKey(path, key_newC1, key_newC2, SNR, u'C_' + str(bit_set_num[k]))

        # readAndWriteDataSet.writeKey(path, key_newF1, key_newF2, SNR, u'F_' + str(bit_set_num[k]))

        # readAndWriteDataSet.writeKey(path, key_old1, key_old2, SNR, u'old_' + str(bit_set_num[k]))

        # readAndWriteDataSet.write(lengths, path + u'length_' + str(bit_set_num[k]))

        # readAndWriteDataSet.write(errorNums, path + u'errorNum_' + str(bit_set_num[k]))

        # readAndWriteDataSet.write(bit_error_rate, path + u'bit_error_rate_' + str(bit_set_num[k]))

    # readAndWriteDataSet.write(bit_error_rate_all, path + u'bit_error_rate_all')

    print("主进程结束！")

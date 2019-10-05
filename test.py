from pylab import *

import readAndWriteDataSet
import tools
import numpy as np
import addNoise
import wt

centroidNum = 1  # 聚类中心数量
iRate = 11  # 预处理后保留的维度
iRate_wt = 18  # 预处理后保留的维度
SNR_initial = 20
column_initial = 1024  # 初始列数
type = u'oushi'
path = u'/Users/jinruimeng/Downloads/keyan/'

if __name__ == '__main__':

    # 读入信道数据
    channelDataPath_a = path + u'channelDataP_a.xlsx'
    # channelDataPath_b = path + u'channelDataP_b.xlsx'
    channelData_a0 = readAndWriteDataSet.excelToMatrixList(channelDataPath_a)
    # channelData_b0 = readAndWriteDataSet.excelToMatrixList(channelDataPath_b)

    # 规范数据
    p = np.shape(channelData_a0)[0]
    channelData_a = []
    channelData_b = []
    for i in range(p):
        channelData_a.append(channelData_a0[i][:, 0:column_initial])
        # channelData_b.append(channelData_b0[i][:, 0:column_initial])
        for j in range(column_initial):
            channelData_a[i][:, j] -= mean(channelData_a[i][:, j])
            # channelData_b[i][:, j] -= mean(channelData_b[i][:, j])
        print("mean", np.mean(channelData_a[i]))

    p, m, n = np.shape(channelData_a)

    covs = tools.getCovMatrixList(channelData_a)
    npowers = addNoise.get_npower(covs)
    print("npowers", npowers)

    for k in range(p):
        print("k", k)

        channelData = channelData_a[k]
        print("shape", np.shape(channelData))

        power = np.linalg.norm(channelData) ** 2
        print("power", power)

        snr = 10 * np.log10((power/m/n/npowers[k]) - 1)
        print("snr", snr)

        cov_pca = np.cov(channelData, rowvar=False)
        U_pca, Sigma, VT = np.linalg.svd(cov_pca)
        result_pca = np.dot(channelData, np.transpose(VT))

        sum = np.sum(Sigma)
        print("sum", sum)

        power_pca = np.linalg.norm(result_pca) ** 2
        print("power_pca", power_pca)

        p = []
        powers_pca = []
        tmpSum = 0
        for i in range(20):
            tmpSum += Sigma[i]
            p.append(tmpSum / sum)
            powers_pca.append((np.linalg.norm(result_pca[:, 0:i + 1]) ** 2) / power_pca)

        result_wt = wt.wt(channelData, np.shape(channelData)[1])
        cov_wt = np.cov(result_wt, rowvar=False)

        sum1 = np.sum(cov_wt)
        print("sum1", sum1)
        power_wt = np.linalg.norm(result_wt) ** 2
        print("power_wt", power_wt)

        p2 = []
        powers_wt = []
        result2 = result_wt[:, 0]
        cov2 = np.cov(result2, rowvar=False)
        sum2 = np.sum(cov2)
        p2.append(sum2 / sum)
        powers_wt.append(np.linalg.norm(result2) ** 2 / power_wt)

        for i in range(1, 20):
            result2 = result_wt[:, 0:i + 1]
            cov2 = np.cov(result2, rowvar=False)
            Sigma2 = np.linalg.svd(cov2)[1]
            sum2 = np.sum(Sigma2)
            p2.append(sum2 / sum)
            powers_wt.append((np.linalg.norm(result2) ** 2) / power_wt)

        print("pca_p", p)
        print("wt_p", p2)
        print("powers_pca", powers_pca)
        print("powers_wt", powers_wt)

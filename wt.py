import readAndWriteDataSet
import tools
import pywt
import numpy as np
import math


def wt(channelData, iRate):
    m, n = np.shape(channelData)
    U = np.ones((n, iRate))
    for j in range(1, iRate):
        a = int(math.log2(j))
        b = j + 1 - (2 ** a)
        for i in range(n):
            if (b - 1) / (2 ** a) < (i + 1) / n <= (b - 0.5) / (2 ** a):
                U[i, j] = 2 ** (a / 2)
            else:
                if (b - 0.5) / (2 ** a) < (i + 1) / n <= b / (2 ** a):
                    U[i, j] = - 2 ** (a / 2)
                else:
                    U[i, j] = 0
    U = U / np.sqrt(n)
    result = np.dot(channelData, U)

    return result

def wt2(channelData, iRate):
    m, n = np.shape(channelData)
    result = channelData

    while 1:
        tmp = []

        if np.shape(result)[1] <= max(2, iRate):
            break

        for i in range(m):
            tmp.append(pywt.dwt(result[i][0:(int(np.shape(result[i])[0] / 2)) * 2], 'haar')[0])
        result = tools.listToArray(tmp)

    return result


# 计算功率
def get_npower(dataList):
    # 单位转换: dB到倍数
    # m, n = np.shape(dataList)
    xpower = (np.linalg.norm(dataList) ** 2)

    return xpower


if __name__ == '__main__':
    type = u'oushi'
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'D:\\Users\\rmjin\\Downloads\\'

    # 读取数据
    channelDataPath = path + "channelDataP_a.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    K = np.shape(channelDataAll)[0]
    for k in range(K):
        print("k", k)

        channelData = channelDataAll[k][:, 0:1024]
        print("shape", np.shape(channelData))

        power = np.linalg.norm(channelData) ** 2
        print("power", power)

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

        result_wt = wt(channelData, np.shape(channelData)[1])
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
        p2.append(sum2 / sum1)
        powers_wt.append(np.linalg.norm(result2) ** 2 / power_wt)

        for i in range(1, 20):
            result2 = result_wt[:, 0:i + 1]
            cov2 = np.cov(result2, rowvar=False)
            Sigma2 = np.linalg.svd(cov2)[1]
            sum2 = np.sum(Sigma2)
            p2.append(sum2 / sum1)
            powers_wt.append((np.linalg.norm(result2) ** 2) / power_wt)

        print("pca_p", p)
        print("wt_p", p2)
        print("powers_pca", powers_pca)
        print("powers_wt", powers_wt)

    # n = iRate = 1024
    # U = np.ones((n, iRate))
    # for j in range(1, iRate):
    #     a = int(math.log2(j))
    #     b = j + 1 - (2 ** a)
    #     for i in range(n):
    #         if (b - 1) / (2 ** a) < (i + 1) / n <= (b - 0.5) / (2 ** a):
    #             U[i, j] = 2 ** (a / 2)
    #         else:
    #             if (b - 0.5) / (2 ** a) < (i + 1) / n <= b / (2 ** a):
    #                 U[i, j] = - 2 ** (a / 2)
    #             else:
    #                 U[i, j] = 0
    # U = U / np.sqrt(n)
    # U2 = np.asarray(np.mat(U).I)
    # U3 = np.asarray(np.mat(U).T)
    # U4 = np.asarray(U2 - U3)
    # sumList = []
    # for j in range(iRate):
    #     sum = 0
    #     for i in range(n):
    #         sum += U[i, j] ** 2
    #     sumList.append(sum)
    print("finish")

import numpy as np


# 获得噪声功率
# def get_npower(dataList, snr):
#     # 单位转换: dB到倍数
#     snr = 10 ** (snr / 10.0)
#     p, m, n = np.shape(dataList)
#     npowers = []
#     for i in range(p):
#         xpower = abs((np.linalg.norm(dataList[i]) ** 2) / (m * n))
#         npowers.append(xpower / (1 + snr))
#
#     return npowers

# 获得噪声功率
def get_npower(covList):
    # 单位转换: dB到倍数
    p, m, n = np.shape(covList)
    npowers = []
    for i in range(p):
        Sigma = np.linalg.svd(covList[i])[1]
        npowers.append(min(Sigma))

    return npowers


# 获得每一列的信噪比
def get_SNR_list(channelData1, channelData2, npower):
    SNRList = []
    m, n = np.shape(channelData1)

    covMatrix1 = np.cov(channelData1, rowvar=False)
    covMatrix2 = np.cov(channelData2, rowvar=False)

    for i in range(n):
        SNR = (max(covMatrix1[i, i].real, covMatrix2[i, i].real) / npower) - 1
        if SNR < 0:
            SNR = 0.01
        SNR = 10 * np.log10(SNR)
        SNRList.append(SNR)

    return SNRList


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'  # path = u'E:\\workspace\\keyan\\'

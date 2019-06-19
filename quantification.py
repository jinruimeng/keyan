import numpy as np


# 量化
# input: channelData1-Alice信道数据 channelData2-Bob信道数据
# output: key1-Alice量化后的密钥 key2-Bob量化后的密钥
def quantificate(channelData1, channelData2):
    m, n = np.shape(channelData1)
    for i in range(m):
        # 本方案对幅度进行量化
        for j in range(n):
            channelData1[i, j] = abs(channelData1[i, j])
            channelData2[i, j] = abs(channelData2[i, j])

    key1 = u''
    key2 = u''
    covMatrix1 = np.cov(channelData1, rowvar=False)
    covMatrix2 = np.cov(channelData2, rowvar=False)

    for i in range(n):
        tmpKey = quantificateWithSNR(channelData1[:, i], channelData2[:, i],
                                     max(covMatrix1[i, i], covMatrix2[i, i]) - 1)
        key1 = key1 + tmpKey[0]
        key2 = key2 + tmpKey[1]

    return key1, key2


def quantificateWithSNR(data1, data2, SNR):
    SNR = 10 * np.log10(SNR)
    if SNR >= 42.3:
        bitNum = 8
    else:
        if SNR >= 36.3:
            bitNum = 6
        else:
            if SNR >= 30.4:
                bitNum = 4
            else:
                if SNR >= 24.5:
                    bitNum = 2
                else:
                    bitNum = 0

    # 寻找最大值和最小值
    maxNum = max(data1.max(), data2.max())
    minNum = min(data1.min(), data2.min())
    # 确定量化间隔
    deta = (maxNum - minNum) / (1 << bitNum)
    # 确定密钥形式，后续可以考虑引入格雷码
    keyFormat = "{0:0" + str(bitNum) + "b}"
    tmpKey1 = u''
    tmpKey2 = u''
    m = np.shape(data1)[0]
    for i in range(m):
        tmpKey1 = tmpKey1 + keyFormat.format((int)((int)((data1[i] - minNum) / deta)-0.001))
        tmpKey2 = tmpKey2 + keyFormat.format((int)((int)((data2[i] - minNum) / deta)-0.001))

    return tmpKey1, tmpKey2


# 计算密钥不一致率
# input: key1-Alice量化后的密钥 key2-Bob量化后的密钥
# output: 密钥不一致率
def getInconsistencyRate(key1, key2):
    keyLength = len(key1)
    if keyLength <= 0:
        return 0
    key1List = list(key1)
    key2List = list(key2)
    num = 0
    for index in range(len(key1List)):
        if key1List[index] is not key2List[index]:
            num += 1

    return num / keyLength

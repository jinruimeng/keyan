import numpy as np


# 量化
# input: channelData1-Alice信道数据 channelData2-Bob信道数据
# output: key1-Alice量化后的密钥 key2-Bob量化后的密钥
def quantificate(channelData1, channelData2, npower):
    SNRList = []
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
    key1List = []
    key2List = []

    for i in range(n):
        SNR = (max(abs(covMatrix1[i, i]), abs(covMatrix2[i, i])) / npower) - 1
        # print(u'i:')
        # print(str(i))
        # print(u'SNR0:')
        # print(SNR)
        SNR = 10 * np.log10(SNR)
        # print(u'SNR1:')
        # print(SNR)
        SNRList.append(SNR)
        tmpKey = quantificateWithSNR(channelData1[:, i], channelData2[:, i], SNR)
        key1List.append(tmpKey[0])
        key2List.append(tmpKey[1])

    for i in range(m):
        try:
            for j in range(n):
                try:
                    key1 = key1 + key1List[j][i]
                    key2 = key2 + key2List[j][i]
                except:
                    continue
        except:
            continue

    return key1, key2, SNRList


def quantificateWithSNR(data1, data2, SNR):
    tmpKey1 = []
    tmpKey2 = []

    if SNR >= 51.1:
        bitNum = 11
    else:
        if SNR >= 48.2:
            bitNum = 10
        else:
            if SNR >= 45.2:
                bitNum = 9
            else:
                if SNR >= 42.3:
                    bitNum = 8
                else:
                    if SNR >= 39.3:
                        bitNum = 7
                    else:
                        if SNR >= 36.3:
                            bitNum = 6
                        else:
                            if SNR >= 33.3:
                                bitNum = 5
                            else:
                                if SNR >= 30.4:
                                    bitNum = 4
                                else:
                                    if SNR >= 27.4:
                                        bitNum = 3
                                    else:
                                        if SNR >= 24.5:
                                            bitNum = 2
                                        else:
                                            if SNR >= 15.4:
                                                bitNum = 1
                                            else:
                                                return tmpKey1, tmpKey2

    # 确定密钥形式，后续可以考虑引入格雷码
    keyFormat = "{0:0" + str(bitNum) + "b}"
    # 寻找最大值和最小值
    maxNum = max(data1.max().real, data2.max().real)
    minNum = min(data1.min().real, data2.min().real)
    # 确定量化间隔
    deta = (maxNum - minNum) / (1 << bitNum)
    m = np.shape(data1)[0]
    for i in range(m):
        tmpKey1.append(keyFormat.format((int)(((data1[i].real - minNum) / deta) - 0.001)))
        tmpKey2.append(keyFormat.format((int)(((data2[i].real - minNum) / deta) - 0.001)))

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

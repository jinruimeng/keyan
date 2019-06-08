import numpy as np

# 量化
# input: channelData1-Alice信道数据 channelData2-Bob信道数据 bitNum-量化比特数
# output: key1-Alice量化后的密钥 key2-Bob量化后的密钥
def quantificate(channelData1, channelData2, bitNum):
    bitNum = bitNum.real
    try:
        # 判断输入是单个信道还是多个信道
        # 如果是单个信道，进行以下步骤
        m, n = np.shape(channelData1)
        for i in range(m):
            # 本方案对幅度进行量化
            for j in range(n):
                channelData1[i, j] = abs(channelData1[i, j])
                channelData2[i, j] = abs(channelData2[i, j])
    except:
        # 如果是多个信道，进行以下步骤
        m = np.shape(channelData1)[0]
        for i in range(m):
            channelData1[i] = abs(channelData1[i])
            channelData2[i] = abs(channelData2[i])

    # 寻找最大值和最小值
    maxNum = max(channelData1.max(), channelData2.max())
    minNum = min(channelData1.min(), channelData2.min())
    # 确定量化间隔
    deta = (int)((maxNum - minNum).real / (1 << bitNum)) + 1
    # 确定密钥形式，后续可以考虑引入格雷码
    keyFormat = "{0:0" + str(bitNum) + "b}"
    key1 = u''
    key2 = u''

    try:
        m, n = np.shape(channelData1)
        for i in range(m):
            for j in range(n):
                key1 = key1 + keyFormat.format((int)(((channelData1[i, j] - minNum) / deta).real))
                key2 = key2 + keyFormat.format((int)(((channelData2[i, j] - minNum) / deta).real))
    except:
        m = np.shape(channelData1)[0]
        for i in range(m):
            key1 = key1 + keyFormat.format((int)(((channelData1[i] - minNum) / deta).real))
            key2 = key2 + keyFormat.format((int)(((channelData2[i] - minNum) / deta).real))

    return key1, key2

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

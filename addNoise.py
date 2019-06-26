import numpy as np
import readAndWriteDataSet
import tools


# 根据信噪比和信道测量值生成高斯白噪声
# input: x-信道数据 snr-信噪比（dB）
# output: noise-高斯白噪声
def wgn(x, snr):
    # 单位转换: dB到倍数
    snr = 10 ** (snr / 10.0)

    try:
        # 如果输入是矩阵，进行以下步骤
        m, n = np.shape(x)
        # 计算输入信号平均功率
        xpower = abs((np.linalg.norm(x) ** 2) / (m * n))
        # 求噪声功率
        npower = xpower / snr
        # 生成高斯白噪声
        noise1 = np.array(np.zeros((m, n)), dtype=complex)
        noise2 = np.array(np.zeros((m, n)), dtype=complex)
        for i in range(m):
            for j in range(n):
                noise1[i, j] = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower / 2)
                noise2[i, j] = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower / 2)
    except:
        print("wgn")
        # 如果输入是列表，进行以下步骤
        m = np.shape(x)[0]
        xpower = abs((np.linalg.norm(x) ** 2) / m)
        npower = xpower / snr
        noise1 = []
        noise2 = []
        for i in range(m):
            tmpNoise1 = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower / 2)
            tmpNoise2 = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower / 2)
            noise1.append(tmpNoise1)
            noise2.append(tmpNoise2)

    return noise1, noise2, npower


def wgn_abs(x, snr):
    # 单位转换: dB到倍数
    snr = 10 ** (snr / 10.0)
    npower = []
    mean = []
    try:
        # 如果输入是矩阵，进行以下步骤
        m, n = np.shape(x)
        # 计算输入信号平均功率
        for i in range(n):
            xpower = abs((np.linalg.norm(x[:, i]) ** 2) / m)
            # 求噪声功率
            npower.append(xpower / snr)
            mean.append(0)

        n_cov = np.zeros((n, n))
        for i in range(n):
            n_cov[i, i] = npower[i]

        noise1 = np.random.multivariate_normal(mean, n_cov, m)
        noise2 = np.random.multivariate_normal(mean, n_cov, m)
    except:
        print("wgn_abs")
        # 如果输入是列表，进行以下步骤
        m = np.shape(x)[0]
        xpower = abs((np.linalg.norm(x) ** 2) / m)
        npower.append(xpower / snr)

        n_cov = np.ones((1, 1)) * npower[0]

        noise1 = np.random.multivariate_normal([0], n_cov, m)
        noise2 = np.random.multivariate_normal([0], n_cov, m)

    return noise1, noise2, np.mean(npower)


if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'

    # 信噪比的上下限
    low = -10
    high = 5
    step = 5
    suffix = str(low) + u'to' + str(high) + u'dB'
    time1 = ((int)((high - low) / step + 1))

    # 读入信道数据
    channelDataPath = path + u'channelDataP.xlsx'
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    p = np.shape(channelDataAll)[0]  # 页数

    # 添加噪声
    channelDataAll1 = []
    channelDataAll2 = []
    for h in range(time1):
        SNR = low + h * step
        for i in range(p):
            noise1, noise2, npower = wgn_abs(channelDataAll[i], SNR)
            channelDataAll1.append(channelDataAll[i] + noise1)
            channelDataAll2.append(channelDataAll[i] + noise2)

    outChannelAll1ListPath = path + "outChannelAll1List_" + suffix
    outChannelAll2ListPath = path + "outChannelAll2List_" + suffix
    readAndWriteDataSet.write(channelDataAll1, outChannelAll1ListPath, ".xlsx")
    readAndWriteDataSet.write(channelDataAll2, outChannelAll2ListPath, ".xlsx")

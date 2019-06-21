import numpy as np
import readAndWriteDataSet


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

    try:
        # 如果输入是矩阵，进行以下步骤
        m, n = np.shape(x)
        # 计算输入信号平均功率
        xpower = abs((np.linalg.norm(x) ** 2) / (m * n))
        # 求噪声功率
        npower = xpower / snr
        # 生成高斯白噪声
        noise1 = np.array(np.zeros((m, n)))
        noise2 = np.array(np.zeros((m, n)))
        for i in range(m):
            for j in range(n):
                noise1[i, j] = np.random.randn() * np.sqrt(npower)
                noise2[i, j] = np.random.randn() * np.sqrt(npower)
    except:
        # 如果输入是列表，进行以下步骤
        m = np.shape(x)[0]
        xpower = abs((np.linalg.norm(x) ** 2) / m)
        npower = xpower / snr
        noise1 = []
        noise2 = []
        for i in range(m):
            tmpNoise1 = np.random.randn() * np.sqrt(npower)
            tmpNoise2 = np.random.randn() * np.sqrt(npower)
            noise1.append(tmpNoise1)
            noise2.append(tmpNoise2)

    return noise1, noise2, npower


if __name__ == '__main__':
    type = u'oushi'
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'

    # 读取数据
    channelDataPath = path + "getCentroids_outCentroidList_U_1_2019-06-04.21.09.48_1.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    noise_V = np.array(np.zeros((2400, 1)))

    for i in range(2400):
        for j in range(2400):
            noise_V[i, 0] += abs(channelDataAll[0][j, i]) ** 2

    out = []
    out.append(noise_V)

    outPath = path + "noise_V.xlsx"
    readAndWriteDataSet.write(out, outPath, ".xlsx")

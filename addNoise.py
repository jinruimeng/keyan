import numpy as np
import readAndWriteDataSet


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)

    # m, n = np.shape(x)
    # xpower = (np.linalg.norm(x) ** 2) / (m * n)
    # npower = xpower / snr
    # noise = np.array(np.zeros((m, n)), dtype=complex)
    # for i in range(m):
    #     for j in range(n):
    #         noise[i, j] = (np.random.randn() + np.random.randn() * j) * np.sqrt(npower)

    try:
        m, n = np.shape(x)
        xpower = (np.linalg.norm(x) ** 2) / (m * n)
        npower = xpower / snr
        noise = np.array(np.zeros((m, n)), dtype=complex)
        for i in range(m):
            for j in range(n):
                noise[i, j] = complex(np.random.randn(),np.random.randn()) * np.sqrt(npower)
    except:
        m = np.shape(np.shape(x))[0]
        xpower = (np.linalg.norm(x) ** 2) / m
        npower = xpower / snr
        noise = []
        for i in range(m):
            tmpNoise = complex(np.random.randn(),np.random.randn()) * np.sqrt(npower)
            noise.append(tmpNoise)

    return noise


if __name__ == '__main__':
    type = u'oushi'
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    a = 2

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    addNoise = []
    for i in range(len(channelDataAll)):
        noise = wgn(channelDataAll[i], 10)
        channelAddNoise = channelDataAll[i] + noise
        addNoise.append(channelAddNoise)

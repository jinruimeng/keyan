import numpy as np
import readAndWriteDataSet
import tools


# 获得噪声功率
def get_npower(dataList, snr):
    # 单位转换: dB到倍数
    snr = 10 ** (snr / 10.0)
    p, m, n = np.shape(dataList)
    xpowers = []
    for i in range(p):
        xpower = abs((np.linalg.norm(dataList[i]) ** 2) / (m * n))
        xpowers.append(xpower / (1 + snr))

    return xpowers

if __name__ == '__main__':
    # 路径配置
    path = u'/Users/jinruimeng/Downloads/keyan/'  # path = u'E:\\workspace\\keyan\\'

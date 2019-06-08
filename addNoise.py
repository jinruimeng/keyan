# coding:utf-8
# 支持中文显示
from pylab import *

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import readAndWriteDataSet
import multiprocessing
import getCovMatrix
import pca

# 根据信噪比和信道测量值生成高斯白噪声
# input: x-信道数据 snr-信噪比（dB）
# output: noise-高斯白噪声
def wgn(x, snr):
    # 单位转换: dB到倍数
    snr = 10 ** (snr / 10.0)

    try:
        # 如果输入是单个信号，进行以下步骤
        m, n = np.shape(x)
        # 计算输入信号平均功率
        xpower = (np.linalg.norm(x) ** 2) / (m * n)
        # 求噪声功率
        npower = xpower / snr
        # 生成高斯白噪声
        noise = np.array(np.zeros((m, n)), dtype=complex)
        for i in range(m):
            for j in range(n):
                noise[i, j] = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower) / np.sqrt(2)
    except:
        # 如果输入是多个信号，进行以下步骤
        m = np.shape(x)[0]
        xpower = (np.linalg.norm(x) ** 2) / m
        npower = xpower / snr
        noise = []
        for i in range(m):
            tmpNoise = complex(np.random.randn(), np.random.randn()) * np.sqrt(npower) / np.sqrt(2)
            noise.append(tmpNoise)

    return noise


if __name__ == '__main__':
    type = u'oushi'
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    a = 0

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    iRate = 5
    # 信噪比的上下限
    low = 5
    high = 20
    step = 5

    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])
    time1 = ((int)((high - low) / step + 1))
    time2 = 1 << a
    schedule[0] = time1 * time2

    oldCorrAll = []
    newCorrAll = []
    for i in range(low, high + 1, step):
        # 显示进度
        schedule[1] += 1
        tmpSchedule = schedule[1]
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分开始！')

        channelDataAll1 = []
        channelDataAll2 = []
        oldCorr = []
        newCorr = []
        for j in range(len(channelDataAll)):
            channelDataAll1.append(wgn(channelDataAll[j], i) + channelDataAll[j])
            channelDataAll2.append(wgn(channelDataAll[j], i) + channelDataAll[j])

        newChannelData1 = pca.pca_general(channelDataAll1, iRate)
        newChannelData2 = pca.pca_general(channelDataAll2, iRate)

        channelDataAll1Array = getCovMatrix.matrixListToMatrix_U(channelDataAll1)
        channelDataAll2Array = getCovMatrix.matrixListToMatrix_U(channelDataAll2)
        newChannelData1Array = getCovMatrix.matrixListToMatrix_U(newChannelData1)
        newChannelData2Array = getCovMatrix.matrixListToMatrix_U(newChannelData2)
        for j in range(len(channelDataAll1)):
            oldCowCor = np.corrcoef(channelDataAll1Array[j, :], channelDataAll2Array[j, :])
            newCowCor = np.corrcoef(newChannelData1Array[j, :], newChannelData2Array[j, :])
            oldCorr.append(abs(oldCowCor[0, 1]))
            newCorr.append(abs(newCowCor[0, 1]))
        oldCorrAll.append(np.mean(oldCorr))
        newCorrAll.append(np.mean(newCorr))

        # 显示进度
        print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(
            schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                      schedule[1] / schedule[0] * 100) + u'！')

    X = range(low, high + 1, step)
    plt.xlabel(u'信噪比（dB）')
    plt.ylabel(u'相关系数')
    plt.plot(X, oldCorrAll, 'k-s', label=u'预处理前')
    plt.plot(X, newCorrAll, 'r-v', label=u'无交互PCA')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    print(u'主进程结束！')

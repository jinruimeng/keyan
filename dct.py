# coding:utf-8
# 支持中文显示
from pylab import *

# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import readAndWriteDataSet
import getCovMatrix
import cv2

if __name__ == '__main__':
    type = u'oushi'
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'D:\\Users\\rmjin\\Downloads\\'

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    channelData = getCovMatrix.getAbs(channelDataAll[0])
    channelData3 = channelData - np.mean(np.mean(channelData))
    channelData2 = channelData3[0, ::2]
    # covMatrix = np.corrcoef(channelData3, rowvar=False)
    Y = cv2.dct(channelData3[0, :])
    Y2 = cv2.dct(channelData2)

    X = range(0, 1200, 1)
    plt.plot(X, channelData2, 'k')
    plt.plot(X, Y2, 'b')
    plt.show()

    print(Y2.shape)

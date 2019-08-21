import readAndWriteDataSet
import tools
import pywt
import numpy as np


def wt(channelData, newDimension=1):
    tmp = []
    m, n = np.shape(channelData)
    for i in range(m):
        tmp.append(pywt.dwt(channelData[i, :], 'haar')[0])
    result = tmp
    tmp = []
    while len(result[0]) > newDimension:
        for i in range(m):
            tmp.append(pywt.dwt(result[i][0:((int)(len(result[0]) / 2)) * 2], 'haar')[0])
        result = tmp
        tmp = []
    return tools.listToArray(result)


if __name__ == '__main__':
    type = u'oushi'
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'D:\\Users\\rmjin\\Downloads\\'

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    channelData = tools.getAbs(channelDataAll[0])

    result = wt(channelData, 5)

    print(np.shape(result))

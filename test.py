import readAndWriteDataSet
import numpy as np

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

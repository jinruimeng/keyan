import numpy as np
import readAndWriteDataSet


# in:list形式的信道数据
# out:list形式的协方差
def getCovMatrixList(matrixs):
    covMatrixList = []
    for i in range(len(matrixs)):
        covMatrix = np.cov(matrixs[i], rowvar=False)
        covMatrixList.append(covMatrix)
    return covMatrixList


def matrixListToMatrix(covMatrixs):
    m, n = np.shape(covMatrixs[0])
    allCovMatrix = np.mat(np.zeros((len(covMatrixs), (int)(0.5 * (n ** 2 + n)))))
    for i in range(len(covMatrixs)):
        curMatrix = covMatrixs[i]
        cur = 0
        for j in range(n):
            for k in range(j, n):
                allCovMatrix[i, cur] = curMatrix[j, k]
                cur += 1
    return allCovMatrix


def matrixToMatrixList(allCovMatrix):
    covMatrixList = []
    # m个协方差矩阵，每个协方差矩阵是p*p的
    m, n = np.shape(allCovMatrix)
    p = (int)(0.5 * ((1 + 8 * n) ** 0.5 - 1))
    for i in range(m):
        curMatrix = np.zeros((p, p))
        curRow = 0
        curCol = 0
        for j in range(n):
            curMatrix[curRow, curCol] = allCovMatrix[i, j]
            curMatrix[curCol, curRow] = allCovMatrix[i, j]
            curCol += 1
            if curCol >= p:
                curRow += 1
                curCol = curRow
        covMatrixList.append(curMatrix)
    return covMatrixList


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test2.xlsx'
    dataSource = readAndWriteDataSet.excelToMatrix(datafile)
    covMatrixList = getCovMatrixList(dataSource)
    allCovMatrix = matrixListToMatrix(covMatrixList)
    print(allCovMatrix)
    print('\n')
    covMatrixs2 = matrixToMatrixList(allCovMatrix)
    print(covMatrixs2)

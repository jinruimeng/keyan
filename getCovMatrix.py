import numpy as np
import readAndWriteDataSet


# in:list形式的信道数据
# out:list形式的协方差
def getCovMatrixList(matrixList):
    covMatrixList = []
    for i in range(len(matrixList)):
        # covMatrix = np.cov(matrixList[i], rowvar=False)
        covMatrix = np.corrcoef(matrixList[i], rowvar=False)
        covMatrixList.append(covMatrix)
    return covMatrixList


# in:list形式的信道数据
# out:list形式的相关系数矩阵
def getCorrMatrixList(matrixList):
    corrMatrixList = []
    for i in range(len(matrixList)):
        corrMatrix = np.corrcoef(matrixList[i], rowvar=False)
        corrMatrixList.append(corrMatrix)
    return corrMatrixList


def matrixListToMatrix(covMatrixList):
    m, n = np.shape(covMatrixList[0])
    allCovMatrix = np.array(np.zeros((len(covMatrixList), (int)(0.5 * (n ** 2 + n)))), dtype=complex)
    for i in range(len(covMatrixList)):
        curMatrix = covMatrixList[i]
        cur = 0
        for j in range(n):
            for k in range(j, n):
                # if curMatrix[j, k] >= 0.1:
                #     allCovMatrix[i, cur] = curMatrix[j, k]
                # else:
                #     allCovMatrix[i, cur] = 0
                allCovMatrix[i, cur] = curMatrix[j, k]
                cur += 1
    return allCovMatrix


def matrixToMatrixList(allCovMatrix):
    covMatrixList = []
    # m个协方差矩阵，每个协方差矩阵是p*p的
    m, n = np.shape(allCovMatrix)
    p = (int)(0.5 * ((1 + 8 * n) ** 0.5 - 1))
    for i in range(m):
        curMatrix = np.array(np.zeros((p, p)), dtype=complex)
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


def getInformations(covMatrixList):
    informations = []
    information = np.array(np.zeros((len(covMatrixList), 1)), dtype=complex)
    for i in range(len(covMatrixList)):
        try:
            U, Sigma, VT = np.linalg.svd(covMatrixList[i], full_matrices=0)
            sum = np.sum(Sigma)
        except:
            sum = covMatrixList[i]
        information[i, 0] = sum
    informations.append(information)
    return informations


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test2.xlsx'
    dataSource = readAndWriteDataSet.excelToMatrix(datafile)
    covMatrixList = getCovMatrixList(dataSource)
    allCovMatrix = matrixListToMatrix(covMatrixList)
    print(allCovMatrix)
    print('\n')
    covMatrixs2 = matrixToMatrixList(allCovMatrix)
    print(covMatrixs2)

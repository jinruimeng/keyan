import numpy as np
import readAndWriteDataSet


# in:list形式的信道数据
# out:list形式的协方差
def getCovMatrixList(matrixList):
    covMatrixList = []
    for i in range(len(matrixList)):
        covMatrix = np.cov(matrixList[i], rowvar=False)
        # covMatrix = np.corrcoef(matrixList[i], rowvar=False)
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


# 一个协方差矩阵变成一行
def matrixListToMatrix(covMatrixList):
    m, n = np.shape(covMatrixList[0])
    allCovMatrix = np.array(np.zeros((len(covMatrixList), (int)(0.5 * (n ** 2 + n)))), dtype=complex)
    for i in range(len(covMatrixList)):
        curMatrix = covMatrixList[i]
        cur = 0
        for j in range(n):
            for k in range(j, n):
                allCovMatrix[i, cur] = curMatrix[j, k]
                cur += 1
    return allCovMatrix


# 单纯的把一个矩阵拉成一行,列先走
def matrixListToMatrix_U(UList):
    try:
        m, n = np.shape(UList[0])
        allU = np.array(np.zeros((len(UList), (int)(m * n))), dtype=complex)
        for i in range(len(UList)):
            curU = UList[i]
            cur = 0
            for k in range(n):
                for j in range(m):
                    allU[i, cur] = curU[j, k]
                    cur += 1
    except:
        print(u'matrixListToMatrix_U')
        m = np.shape(UList[0])[0]
        allU = np.array(np.zeros((len(UList), (int)(m))), dtype=complex)
        for i in range(len(UList)):
            curU = UList[i]
            cur = 0
            for j in range(m):
                allU[i, cur] = curU[j]
                cur += 1
    return allU


# 一行变成一个协方差矩阵
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
            curMatrix[curCol, curRow] = allCovMatrix[i, j].conjugate()
            curCol += 1
            if curCol >= p:
                curRow += 1
                curCol = curRow
        covMatrixList.append(curMatrix)
    return covMatrixList


# 单纯的把一行拉成一个矩阵,列先走
def matrixToMatrixList_U(allU):
    UList = []
    # m个变换矩阵，每个变换矩阵是p*p的
    m, n = np.shape(allU)
    p = (int)(n ** 0.5)
    for i in range(m):
        U = np.array(np.zeros((p, p)), dtype=complex)
        for j in range(p):
            U[:, j] = allU[i, j * p:(j + 1) * p]
        UList.append(U)
    return UList


# 根据协方差矩阵列表，获得信息量、权重列表、变换矩阵列表
def getInformations(covMatrixList):
    informations = []
    SigmaList = []
    information = np.array(np.zeros((len(covMatrixList), 1)), dtype=complex)
    UList = []
    for i in range(len(covMatrixList)):
        try:
            U, Sigma, VT = np.linalg.svd(covMatrixList[i])
            UList.append(np.transpose(VT))
            sum = np.sum(Sigma)
            # 将SigmaList中的值换成权重
            for j in range(len(Sigma)):
                Sigma[j] = Sigma[j] / sum
        except:
            print(u'getInformations')
            UList.append((np.ones((1, 1))))
            sum = covMatrixList[i]
            Sigma = np.ones((1, 1))
        information[i, 0] = sum
        SigmaList.append(Sigma)
    informations.append(information)
    return informations, SigmaList, UList


# 输入复数矩阵，获得幅度矩阵
def getAbs(matrix):
    try:
        m, n = np.shape(matrix)
        result = np.array(np.zeros((m, n)), dtype='float32')
        for i in range(m):
            for j in range(n):
                result[i, j] = np.abs(matrix[i, j])
    except:
        print(u'getAbs')
        m = np.shape(matrix)[0]
        result = []
        for i in range(m):
            tmpAbs = np.abs(matrix[i])
            tmpAbs.dtype = 'float32'
            result.append(tmpAbs)
    return result


# 二维列表到二维数组
def listToArray(listObj):
    m = len(listObj)
    n = len(listObj[0])
    out = np.array(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            try:
                out[i, j] = listObj[i][j]
            except:
                print(u'listToArray')
                out[i, j] = 0
    return out


if __name__ == '__main__':
    a = []
    b = []
    # b.append(1)
    # b.append(2)
    # a.append(b)
    c = np.array(np.zeros((2, 2)))
    a.append(c)
    print(np.shape(a))
    print(len(a))

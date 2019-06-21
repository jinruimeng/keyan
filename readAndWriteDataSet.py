from typing import List, Any
import xlsxwriter
import numpy as np
import xlrd
import os
import tools


# 每个工作簿作为一组数据，放到list中
def excelToMatrixList(path):
    wb = xlrd.open_workbook(path)
    sheets: List[Any] = wb.sheets()
    matrixList = []
    for k in range(len(sheets)):
        table = sheets[k]  # 获取一个sheet表
        row = table.nrows  # 行数
        col = table.ncols  # 列数
        matrix = np.array(np.zeros((row, col)), dtype=complex)
        for x in range(row):
            rows = np.matrix(table.row_values(x))
            matrix[x, :] = rows
        matrixList.append(matrix)
    return matrixList


# 每个工作簿作为一行数据，放到一个矩阵中
def excelToMatrix(path):
    wb = xlrd.open_workbook(path)
    sheets: List[Any] = wb.sheets()
    matrix = np.array(np.zeros((len(sheets), sheets[0].nrows * sheets[0].ncols)), dtype=complex)
    for k in range(len(sheets)):
        table = sheets[k]  # 获取第一个sheet表
        row = table.nrows  # 行数
        col = table.ncols  # 列数

        for x in range(row):
            rows = np.matrix(table.row_values(x))
            matrix[k, x * col:(x + 1) * col] = rows
    return matrix


# 读取第一个工作薄，放到一个矩阵中
def excelToMatrix2(path, k=0):
    wb = xlrd.open_workbook(path)
    sheets: List[Any] = wb.sheets()
    table = sheets[k]
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    matrix = np.array(np.zeros((row, col)), dtype=complex)  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        matrix[:, x] = cols  # 按列把数据存进矩阵中
    return matrix


# 将数据存储成excel
def write(outData, path, suffix=".xlsx"):
    if ".xlsx" == suffix:
        workbook = xlsxwriter.Workbook(path + "_1" + suffix)  # 创建一个Excel文件
        try:
            h, l = np.shape(outData)
            worksheet = workbook.add_worksheet()  # 创建sheet
            for j in range(h):
                for k in range(l):
                    tmp = str(outData[j, k])
                    worksheet.write_string(j, k, tmp)
        except:
            m = len(outData)
            try:
                [h, l] = outData[0].shape
                buff = 0
                bn = 2
                for i in range(m):
                    if buff >= 1800000:
                        workbook.close()
                        workbook = xlsxwriter.Workbook(path + "_" + str(bn) + suffix)
                        buff = 0
                        bn += 1
                    [h, l] = outData[i].shape
                    worksheet = workbook.add_worksheet()  # 创建sheet
                    for j in range(h):
                        for k in range(l):
                            tmp = str(outData[i][j, k])
                            worksheet.write_string(j, k, tmp)
                            buff += 1
            except:
                workbook = xlsxwriter.Workbook(path + "_1" + suffix)  # 创建一个Excel文件
                worksheet = workbook.add_worksheet()  # 创建sheet
                for i in range(m):
                    tmp = str(outData[i])
                    worksheet.write_string(i, 1, tmp)
        workbook.close()
    if ".npy" == suffix:
        np.save(path, outData)


def readCentroids(path, iRate, type, a):
    allCentroids = []
    allCentroidUList = []
    for g in range(1, (1 << a) + 1):
        # 读取聚类中心
        centroidListPath = path + u'getCentroids_outCentroidList_' + type + u'_' + str(g) + u'_'
        # 合并多个文件
        centroidList_g = []
        UList_g = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                file = os.path.join(root, file)
                if centroidListPath in file:
                    centroidListTmp = excelToMatrixList(file)
                    for centroid in centroidListTmp:
                        centroidList_g.append(centroid)
            break

        # 计算聚类中心的变换矩阵
        if u'C' == type:
            for i in range(len(centroidList_g)):
                U, Sigma, VT = np.linalg.svd(centroidList_g[i])
                sum = np.sum(Sigma)
                curSum = 0
                if iRate <= 1:
                    index = 0
                    for j in range(len(Sigma)):
                        curSum += Sigma[j]
                        if iRate - (curSum / sum) > 0:
                            index += 1
                        else:
                            break
                else:
                    index = iRate - 1
                U2 = np.transpose(VT[0:index + 1, :])
                UList_g.append(U2)
            allCentroids.append(tools.matrixListToMatrix(centroidList_g))
            allCentroidUList.append(UList_g)

        if u'U' == type:
            for i in range(len(centroidList_g)):
                U2 = centroidList_g[i][:, 0:iRate]
                for j in range(np.shape(U2)[1]):
                    # 噪声功率归一
                    U2[:, j] = U2[:, j] / np.linalg.norm((U2[:, j]))
                UList_g.append(U2)
            allCentroids.append(tools.matrixListToMatrix_U(centroidList_g))
            allCentroidUList.append(UList_g)

    return allCentroids, allCentroidUList


# 二维列表到二维数组
def listToArray(listObj):
    m = len(listObj)
    n = len(listObj[0])
    out = np.array(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = listObj[i][j]

    return out

# 将密钥输出到txt中
def writeKey(path, keys1, keys2, SNR, type):
    name1 = path + u'/key1_' + str(SNR) + u'_' + type + u'.txt'
    name2 = path + u'/key2_' + str(SNR) + u'_' + type + u'.txt'
    data = open(name1, 'w+')
    for key in keys1:
        print(key, file=data)
    data.close()

    data = open(name2, 'w+')
    for key in keys2:
        print(key, file=data)
    data.close()


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(excelToMatrixList(datafile))
    # str = "1+2j"
    # com = complex(str)
    # print(com)

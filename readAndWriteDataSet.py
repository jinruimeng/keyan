from typing import List, Any
import xlsxwriter
import numpy as np
import xlrd


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
        m = len(outData)
        buff = 0
        bn = 1
        workbook = xlsxwriter.Workbook(path + "_0" + suffix)  # 创建一个Excel文件
        for i in range(m):
            if buff > 10000:
                workbook.close()
                workbook = xlrd.open_workbook(path + "_" + str(bn) + suffix)
                buff = 0
                bn += 1
            [h, l] = outData[i].shape
            worksheet = workbook.add_worksheet()  # 创建sheet
            for j in range(h):
                for k in range(l):
                    tmp = str(outData[i][j, k])
                    worksheet.write_string(j, k, tmp)
                    # buff += 1
        workbook.close()
    if ".npy" == suffix:
        np.save(path, outData)


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(excelToMatrixList(datafile))
    # str = "1+2j"
    # com = complex(str)
    # print(com)

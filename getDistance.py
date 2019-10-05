import numpy as np
import math
import readAndWriteDataSet
import copy
from sklearn import metrics


# 欧氏距离计算
# numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False 求一个数组中给定轴上的元素的总和
# **代表乘方
def oushiDistance(x, y):
    # sum = 0
    # for i in range(len(x)):
    #     if i > 0 and ((abs(x[i]) - abs(x[i - 1])) * (abs(y[i]) - abs(y[i - 1])) < 0):
    #             sum += abs(x[i] - y[i]) * 5
    #     else:
    #         sum += abs(x[i] - y[i])

    # sum = np.linalg.norm((x - y))
    # return sum

    # 计算相关系数的绝对值
    corr = - abs(np.corrcoef(x, y)[0, 1])
    return corr


# 输入为实际信道数据，有噪声
# def costFunction(x, U, npower):
#     x2 = np.dot(x, U)
#     c1 = np.cov(x2, rowvar=False)
#     c2 = copy.deepcopy(c1)
#     for i in range(np.shape(c2)[0]):
#         for j in range(np.shape(c2)[1]):
#             c2[i, j] = max(c2[i, j] - npower, 0)
#     a1 = np.linalg.det(c1)
#     a2 = np.linalg.det(c1 - np.dot(np.dot(c2, np.linalg.inv(c1)), c2))
#     if a2 < 0:
#         a2 = 0
#     y = a1 / a2
#     I = math.log(y, 2)
#     return I


# 输入为实际信道数据，有噪声
def costFunction(x1, x2, U, npower):
    y1 = np.dot(x1, U)
    y2 = np.dot(x2, U)
    # U, Sigma, VT = np.linalg.svd(np.cov(y1, y2, rowvar=False))
    # I = 0
    # for i in range(len(Sigma)):
    #     tmp_npower = min(npower, Sigma[i])
    #     A = ((Sigma[i] - tmp_npower) / Sigma[i]) ** 2
    #     B = (1 - A) ** (-1)
    #     I += math.log(B, 2)
    z1 = []
    z2 = []
    for i in range(np.shape(y1)[0]):
        for j in range(np.shape(y1)[1]):
            z1.append(y1[i, j])
            z2.append(y2[i, j])
    I = abs(metrics.normalized_mutual_info_score(z1, z2, 'arithmetic'))
    return I

def get_normalized_mutual_info_score(x1, x2):
    z1 = []
    z2 = []
    for i in range(np.shape(x1)[0]):
        for j in range(np.shape(x1)[1]):
            z1.append(x1[i, j])
            z2.append(x2[i, j])
    I = abs(metrics.normalized_mutual_info_score(z1, z2, 'arithmetic'))
    return I


if __name__ == '__main__':
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    z = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cov = np.cov(z, rowvar=False)
    print(cov)

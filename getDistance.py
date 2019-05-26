import numpy as np
import math


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

    sum = np.linalg.norm((x - y))
    return sum
    # return abs(np.corrcoef(x, y))  # 计算相关系数的绝对值

# 马氏距离计算
def mashiDistance(x, y, IcovMatrix):
    dif = x - y
    distance = abs(np.dot(dif, np.dot(IcovMatrix, np.transpose(dif))))
    return distance


if __name__ == '__main__':
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    z = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cov = np.cov(z, rowvar=False)
    print(cov)

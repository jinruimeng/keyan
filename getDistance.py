import numpy as np
import math


# 欧氏距离计算
# numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False 求一个数组中给定轴上的元素的总和
# **代表乘方
def oushiDistance(x, y):
    return np.sum(np.square(x - y))  # 计算欧氏距离的平方


def mashiDistance(x, y, IcovMatrix):
    dif = x - y
    distance = math.fabs(np.dot(dif, np.dot(IcovMatrix, np.transpose(dif))))
    return distance


if __name__ == '__main__':
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    z = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cov = np.cov(z, rowvar=False)
    print(cov)

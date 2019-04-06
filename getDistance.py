import numpy as np


# 欧氏距离计算
# numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False 求一个数组中给定轴上的元素的总和
# **代表乘方
def oushiDistance(x, y):
    return np.sum(np.square(x - y))  # 计算欧氏距离的平方


def mashiDistance(x, y):
    print(x)
    print(y)
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])

    print(X)
    XT = X.T

    print(XT)

    # 方法一：根据公式求解
    S = np.cov(X)  # 两个维度之间协方差矩阵
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    # 马氏距离计算两个样本之间的距离，此处共有4个样本，两两组合，共有6个距离。
    n = XT.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            print(d)
            d1.append(d)


if __name__ == '__main__':
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    z = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cov = np.cov(z, rowvar=False)
    print(cov)

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
    return sum  # return abs(np.corrcoef(x, y))  # 计算相关系数的绝对值

# 输入为理想信道数据，无噪声
def costFunction(x, U, npower):
    x2 = np.dot(x, U)
    U, Sigma, VT = np.linalg.svd(np.cov(x2, rowvar=False))
    I = 0
    for i in range(len(Sigma)):
        A = (1 + npower / Sigma[i]) ** (-2)
        B = (1 - A) ** (-1)
        I += math.log(B, 2)
    return I

# 输入为实际信道数据，有噪声
def costFunction2(x, U, npower):
    x2 = np.dot(x, U)
    U, Sigma, VT = np.linalg.svd(np.cov(x2, rowvar=False))
    I = 0
    for i in range(len(Sigma)):
        A = (1 + npower / Sigma[i] - npower) ** (-2)
        B = (1 - A) ** (-1)
        I += math.log(B, 2)
    return I

if __name__ == '__main__':
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    z = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cov = np.cov(z, rowvar=False)
    print(cov)

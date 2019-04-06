import kmeans
import numpy as np
import readAndWriteDataSet
import matplotlib.pyplot as plt


def elbow(dataSet, low, high, type="oushi"):
    m = np.shape(dataSet)[0]  # 行的数目

    '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和

    for k in range(low, high):
        if "oushi" == type:
            centroids, clusterAssment = kmeans.KMeansOushi(dataSet, k)
        else:
            centroids, clusterAssment = kmeans.KMeansMashi(dataSet, k)
        sum = 0
        for i in range(m):
            sum += clusterAssment[i, 1]
        SSE.append(sum)

    X = range(low, high)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    m = np.shape(dataSet)[0]  # 行的数目
    n = np.shape(dataSet)[1]  # 数据维度
    low = 1
    high = m >> 1

    '利用SSE选择k'
    SSE1 = []  # 存放每次结果的误差平方和
    SSE2 = []  # 存放每次结果的误差平方和

    for k in range(low, high):
        centroids1, clusterAssment1 = kmeans.KMeansOushi(dataSet, k)
        # centroids2, clusterAssment2 = kmeans.KMeansMashi(dataSet, k)
        sum1 = 0
        # sum2 = 0
        for i in range(m):
            sum1 += clusterAssment1[i, 1]
            # sum2 += clusterAssment1[i, 2]
        SSE1.append(sum1)
        # SSE2.append(sum2)

    X = range(low, high)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE1, 'o-')
    # plt.plot(X, SSE2, 'v-')
    plt.show()

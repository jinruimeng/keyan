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
            sum += abs(clusterAssment[i, 1])
        SSE.append(sum)

    X = range(low, high)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()

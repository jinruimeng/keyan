import matplotlib.pyplot as plt
import readAndWriteDataSet
import multiprocessing
import numpy as np
import getCovMatrix
import kmeans
import pca


def elbow(channelDataAll, low, high, step, a, iRate, type="oushi"):
    '利用SSE选择k'
    SSE = []  # 存放每次结果
    ps = multiprocessing.Pool(4)
    for i in range(low, high + 1, step):
        SSE.append(ps.apply_async(elbowCore, args=(channelDataAll, i, a, iRate, type)).get())
        # SSE.append(elbowCore(channelDataAll, i, a, iRate, type))

    ps.close()
    ps.join()
    X = range(low, high + 1, step)
    plt.xlabel('k')
    plt.ylabel("信息量保留")
    plt.plot(X, SSE, 'o-')
    plt.show()
    print("主进程结束！")


def elbowCore(channelDataAll, i, a, iRate, type="oushi"):
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    sub = n >> a
    rates = []

    for g in range(1 << a):
        channelData = []
        for h in range(p):
            channelDataPage = channelDataAll[h]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])

        covMatrixList = getCovMatrix.getCovMatrixList(channelData)
        allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

        # 对协方差进行聚类
        if "mashi" == type:
            centroids, clusterAssment = kmeans.KMeansMashi(allCovMatrix, i)
        else:
            centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, i)
        centroidList = getCovMatrix.matrixToMatrixList(centroids)

        # 分析PCA效果,计算信息量保留程度
        tmpRates = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, iRate)[3][0][:, 1]

        rates.append(np.mean(tmpRates))
        # rates.append(min(tmpRates))

        # count = 0
        # for i in range(len(tmpRates)):
        #     if tmpRates[i] < 0.8:
        #         count += 1
        # rates.append(count)

    rate = np.mean(rates)
    # rate = np.min(rates)

    return rate.real


if __name__ == '__main__':
    type = "oushi"
    iRate = 20
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    a = 3
    elbow(channelDataAll, 1, 51, 5, a, iRate, type="oushi")

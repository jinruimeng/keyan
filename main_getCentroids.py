import multiprocessing
import readAndWriteDataSet
import getCovMatrix
import kmeans
import pca
import elbow
import time
import numpy as np


def getCentroids(type, path, suffix, channelData, g, k):
    print("第" + str(g) + "轮开始！")
    # 得到协方差的集合，并把协方差放到一个矩阵中
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = path + "getCentroids_outOldCovMatrixList_" + type + "_" + str(g) + str(
        nowTime)
    outCentroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_" + str(nowTime)
    outClusterAssmentPath = path + "getCentroids_outClusterAssment_" + type + "_" + str(g) + "_" + str(
        nowTime)
    outNewChannelDataPath = path + "getCentroids_outNewChannelData_" + type + "_" + str(g) + "_" + str(
        nowTime)
    outNewCovMatrixListPath = path + "getCentroids_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(
        nowTime)
    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)

    # 利用肘部法选择聚类中心个数
    # elbow.elbow(allCovMatrix, 1, 8, type)

    # 对协方差进行聚类
    if "mashi" == type:
        centroids, clusterAssment = kmeans.KMeansMashi(allCovMatrix, k)
    else:
        centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, k)

    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)
    centroidList = getCovMatrix.matrixToMatrixList(centroids)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)
    readAndWriteDataSet.write(centroidList, outCentroidListPath, suffix)

    # 分析PCA效果
    newChannelData, newCovMatrixList = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, 1)

    # 输出
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixListPath, suffix)
    print("第" + str(g) + "轮结束！")


if __name__ == '__main__':
    type = "oushi"
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"
    suffix = ".xlsx"

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = path + "getCentroids_outOldCovMatrixList_" + type + "_" + str(
        nowTime) + suffix

    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    a = 5  # 拆分成2^a份
    sub = n >> a
    k = 3  # 聚类中心数量
    ps = multiprocessing.Pool(4)

    for g in range(1 << a):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])
        ps.apply_async(getCentroids, args=(type, path, suffix, channelData, g, k))
        # getCentroids(type, path, suffix, channelData, g, k)

    ps.close()
    ps.join()
    print("主进程结束！")

import readAndWriteDataSet
import kmeans
import getCovMatrix
import pca
import time
import numpy as np
import multiprocessing


def cluster(type, path, suffix, channelData, g):
    print("第" + str(g) + "轮开始！")
    centroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_.xlsx"
    centroidList = readAndWriteDataSet.excelToMatrixList(centroidListPath)

    # 计算出信道数据的协方差
    centroids = getCovMatrix.matrixListToMatrix(centroidList)
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    # 确定每个数据分别属于哪个簇
    clusterAssment = kmeans.getClusterAssment(allCovMatrix, centroids, type)
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)

    # 分析PCA效果
    newChannelData, newCovMatrixs = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, 0.8)

    # 输出
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = path + "cluster_outOldCovMatrixList_" + type + "_" + str(g) + "_" + str(
        nowTime) + suffix
    outClusterAssmentPath = path + "cluster_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime) + suffix
    outNewChannelDataPath = path + "cluster_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime) + suffix
    outNewCovMatrixsPath = path + "cluster_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime) + suffix
    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath)
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath)
    readAndWriteDataSet.write(newCovMatrixs, outNewCovMatrixsPath)
    print("第" + str(g) + "轮结束！")


if __name__ == '__main__':
    type = "oushi"
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"
    suffix = ".xlsx"
    channelDataPath = path + "channelData1.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    a = 1  # 拆分成2^a份
    sub = n >> a
    ps = multiprocessing.Pool(8)

    for g in range(1 << a):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])
        ps.apply_async(cluster, args=(type, path, suffix, channelData, g))

    ps.close()
    ps.join()
    print("主进程结束！")

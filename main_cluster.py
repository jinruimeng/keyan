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
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCorrMatrixListPath = path + "cluster_outOldCorrMatrixList_" + type + "_" + str(g) + "_" + str(
        nowTime)
    outClusterAssmentPath = path + "cluster_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewChannelDataPath = path + "cluster_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewCovMatrixsPath = path + "cluster_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)

    # 读入聚类中心信息
    centroidList = readAndWriteDataSet.excelToMatrixList(centroidListPath)
    centroids = getCovMatrix.matrixListToMatrix(centroidList)

    # 计算信道相关系数矩阵,并输出
    corrMatrixList = getCovMatrix.getCorrMatrixList(channelData)
    readAndWriteDataSet.write(corrMatrixList, outOldCorrMatrixListPath, suffix)

    # 计算信道数据的协方差
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    # 确定每个数据分别属于哪个簇
    clusterAssment = kmeans.getClusterAssment(allCovMatrix, centroids, type)
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)

    # 输出聚类结果
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)

    # 分析PCA效果
    newChannelData, newCovMatrixs = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, 1)

    # 输出PCA结果
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixs, outNewCovMatrixsPath, suffix)
    print("第" + str(g) + "轮结束！")


if __name__ == '__main__':
    type = "oushi"
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"
    suffix = ".xlsx"
    channelDataPath = path + "channelDataC.xlsx"
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

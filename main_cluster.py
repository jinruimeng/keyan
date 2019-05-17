import readAndWriteDataSet
import kmeans
import getCovMatrix
import pca
import time
import numpy as np
import multiprocessing
import os


def cluster(type, path, suffix, channelData, g, iRate):
    print("第" + str(g) + "轮开始！")
    centroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_"
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = path + "cluster_outOldCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    # outOldInformationsPath = path + "cluster_outOldInformations_" + type + "_" + str(g) + "_" + str(nowTime)
    outClusterAssmentPath = path + "cluster_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewChannelDataPath = path + "cluster_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewCovMatrixsPath = path + "cluster_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    # outNewInformationsPath = path + "cluster_outNewInformation_" + type + "_" + str(g) + "_" + str(nowTime)
    ratesPath = path + "cluster_rates_" + type + "_" + str(g) + "_" + str(nowTime)
    VTsPath = path + "cluster_VTs_" + type + "_" + str(g) + "_" + str(nowTime)

    # 读入聚类中心信息
    centroidList = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            file = os.path.join(root, file)
            if centroidListPath in file:
                centroidListTmp = readAndWriteDataSet.excelToMatrixList(file)
                for centroid in centroidListTmp:
                    centroidList.append(centroid)
        break
    centroids = getCovMatrix.matrixListToMatrix(centroidList)

    # 计算信道相关系数矩阵并输出，然后放到一个矩阵中
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    # oldInformations = getCovMatrix.getInformations(covMatrixList)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)
    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)
    # readAndWriteDataSet.write(oldInformations, outOldInformationsPath, suffix)

    # 确定每个数据分别属于哪个簇
    clusterAssment = kmeans.getClusterAssment(allCovMatrix, centroids, type)
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)

    # 输出聚类结果
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)

    # 分析PCA效果
    newChannelData, newCovMatrixList, VTs, rates = pca.pca(channelData, covMatrixList, centroidList, clusterAssment,
                                                           iRate)
    # newInformations = getCovMatrix.getInformations(newCovMatrixList)

    # 输出PCA结果
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixsPath, suffix)
    # readAndWriteDataSet.write(newInformations, outNewInformationsPath, suffix)
    readAndWriteDataSet.write(VTs, VTsPath, suffix)
    readAndWriteDataSet.write(rates, ratesPath, suffix)
    print("第" + str(g) + "轮结束！")


if __name__ == '__main__':
    type = "oushi"
    # path = "/Users/jinruimeng/Downloads/keyan/"
    path = "E:\\workspace\\keyan\\"
    suffix = ".xlsx"
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    ps = multiprocessing.Pool(4)
    a = 2  # 拆分成2^a份
    sub = n >> a
    iRate = 5

    for g in range(1 << a):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])
        ps.apply_async(cluster, args=(type, path, suffix, channelData, g, iRate))
        # cluster(type, path, suffix, channelData, g, iRate)
    ps.close()
    ps.join()
    print("主进程结束！")

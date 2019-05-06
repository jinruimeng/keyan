import multiprocessing
import readAndWriteDataSet
import getCovMatrix
import kmeans
import pca
import time
import numpy as np


def getCentroids(type, path, suffix, channelData, g, k, iRate):
    print("第" + str(g) + "轮开始！")
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = path + "getCentroids_outOldCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    # outOldInformationsPath = path + "getCentroids_outOldInformations_" + type + "_" + str(g) + "_" + str(nowTime)
    outCentroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_" + str(nowTime)
    outClusterAssmentPath = path + "getCentroids_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewChannelDataPath = path + "getCentroids_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewCovMatrixListPath = path + "getCentroids_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    # outNewInformationsPath = path + "getCentroids_outNewInformation_" + type + "_" + str(g) + "_" + str(nowTime)
    ratesPath = path + "getCentroids_rates_" + type + "_" + str(g) + "_" + str(nowTime)
    VTsPath = path + "getCentroids_VTs_" + type + "_" + str(g) + "_" + str(nowTime)

    # 得到相关系数矩阵并输出,然后放到一个矩阵中
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)
    # oldInformations = getCovMatrix.getInformations(covMatrixList)
    # readAndWriteDataSet.write(oldInformations, outOldInformationsPath, suffix)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    # 对协方差进行聚类
    if "mashi" == type:
        centroids, clusterAssment = kmeans.KMeansMashi(allCovMatrix, k)
    else:
        centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, k)

    # 输出聚类结果
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)
    centroidList = getCovMatrix.matrixToMatrixList(centroids)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)
    readAndWriteDataSet.write(centroidList, outCentroidListPath, suffix)

    # 分析PCA效果
    newChannelData, newCovMatrixList, VTs, rates = pca.pca(channelData, covMatrixList, centroidList, clusterAssment,
                                                           iRate)
    # newInformations = getCovMatrix.getInformations(newCovMatrixList)

    # 输出PCA后结果
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixListPath, suffix)
    # readAndWriteDataSet.write(newInformations, outNewInformationsPath, suffix)
    readAndWriteDataSet.write(VTs, VTsPath, suffix)
    readAndWriteDataSet.write(rates, ratesPath, suffix)
    print("第" + str(g) + "轮结束！")


if __name__ == '__main__':
    type = "oushi"
    path = "/Users/jinruimeng/Downloads/keyan/"
    # path = "E:\\workspace\\keyan\\"
    suffix = ".xlsx"

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    ps = multiprocessing.Pool(4)
    a = 5  # 拆分成2^a份
    sub = n >> a
    k = 1  # 聚类中心数量
    iRate = 0.9

    for g in range(1 << a):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, g * sub:(g + 1) * sub])
        ps.apply_async(getCentroids, args=(type, path, suffix, channelData, g, k, iRate))
        # getCentroids(type, path, suffix, channelData, g, k,iRate)

    ps.close()
    ps.join()
    print("主进程结束！")

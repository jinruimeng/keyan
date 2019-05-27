import readAndWriteDataSet
import kmeans
import getCovMatrix
import pca
import time
import numpy as np
import multiprocessing
import os


def cluster(schedule, type, path, suffix, channelData, g, iRate):
    if iRate > np.shape(channelData)[1]:
        print(u'降维后维度不能大于样本原有的维度！')
        return
    if iRate <= 0:
        print(u'降维后维度不能小于1！')
        return

    print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(g) + u'部分开始！')
    centroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_"
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    # outOldCovMatrixListPath = path + "cluster_outOldCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    outClusterAssmentPath = path + "cluster_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewChannelDataPath = path + "cluster_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewCovMatrixsPath = path + "cluster_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    ratesPath = path + "cluster_rates_" + type + "_" + str(g) + "_" + str(nowTime)
    UTsPath = path + "cluster_UTs_" + type + "_" + str(g) + "_" + str(nowTime)

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
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)
    # 协方差矩阵太大了，先不输出
    # readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)

    # 确定每个数据分别属于哪个簇
    clusterAssment = kmeans.getClusterAssment(allCovMatrix, centroids, type)

    # 输出聚类结果
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)

    # 分析PCA效果
    newChannelData, newCovMatrixList, UTs, rates = pca.pca(channelData, covMatrixList, centroidList, clusterAssment,
                                                           iRate)

    # 输出PCA结果
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixsPath, suffix)
    readAndWriteDataSet.write(UTs, UTsPath, suffix)
    readAndWriteDataSet.write(rates, ratesPath, suffix)

    # 显示进度
    schedule[1] += 1
    print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(g) + u'部分完成，' + u'已完成' + str(
        schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                  schedule[1] / schedule[0] * 100) + u'！')


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])
    type = u'oushi'
    path = u'/Users/jinruimeng/Downloads/keyan/'
    # path = u'E:\\workspace\\keyan\\'
    suffix = u'.xlsx'
    channelDataPath = path + u'channelDataP.xlsx'
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    ps = multiprocessing.Pool(4)
    a = 0  # 拆分成2^a份
    schedule[0] = 1 << a
    sub = n >> a
    iRate = 4  # 降维后维度

    for g in range(1, (1 << a) + 1):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, (g - 1) * sub:g * sub])
        ps.apply_async(cluster, args=(schedule, type, path, suffix, channelData, g, iRate))
        # cluster(schedule, type, path, suffix, channelData, g, iRate)
    ps.close()
    ps.join()
    print("主进程结束！")

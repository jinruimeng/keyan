import multiprocessing
import readAndWriteDataSet
import getCovMatrix
import kmeans
import pca
import time
import numpy as np


def getCentroids(schedule, type, path, suffix, channelData, g, k, iRate):
    # 校验数据正确性
    if k > np.shape(channelData)[0]:
        print(u'聚类中心数量不能大于样本数量！')
        return
    if iRate > np.shape(channelData)[1]:
        print(u'降维后维度不能大于样本原有的维度！')
        return
    if k <= 0 or iRate <= 0:
        print(u'聚类中心数量和降维后维度不能小于1！')
        return

    print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(g) + u'部分开始！')
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    # outOldCovMatrixListPath = path + "getCentroids_outOldCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    # outCentroidListPath = path + "getCentroids_outCentroidList_" + type + "_" + str(g) + "_" + str(nowTime)
    outClusterAssmentPath = path + "getCentroids_outClusterAssment_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewChannelDataPath = path + "getCentroids_outNewChannelData_" + type + "_" + str(g) + "_" + str(nowTime)
    outNewCovMatrixListPath = path + "getCentroids_outNewCovMatrixList_" + type + "_" + str(g) + "_" + str(nowTime)
    ratesPath = path + "getCentroids_rates_" + type + "_" + str(g) + "_" + str(nowTime)
    UTsPath = path + "getCentroids_UTs_" + type + "_" + str(g) + "_" + str(nowTime)

    # 得到相关系数矩阵并输出,然后放到一个矩阵中
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    # 协方差矩阵太大了，先不输出
    # readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)
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
    # 聚类中心太大了，先不输出
    # readAndWriteDataSet.write(centroidList, outCentroidListPath, suffix)

    # 分析PCA效果
    informations = getCovMatrix.getInformations(covMatrixList)[0]
    newChannelData, newCovMatrixList, UTs, rates = pca.pca(channelData, informations, centroidList, clusterAssment,
                                                           iRate)

    # 输出PCA后结果
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixListPath, suffix)
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

    # 读取数据
    channelDataPath = path + u'channelDataP.xlsx'
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)

    n = np.shape(channelDataAll[0])[1]  # 列数
    p = len(channelDataAll)  # 页数
    ps = multiprocessing.Pool(4)
    a = 0  # 拆分成2^a份
    schedule[0] = 1 << a
    sub = n >> a
    iRate = 4  # 降维后维度
    k = 7  # 聚类中心数量

    for g in range(1, (1 << a) + 1):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, (g - 1) * sub:g * sub])
        ps.apply_async(getCentroids, args=(schedule, type, path, suffix, channelData, g, k, iRate))
        # getCentroids(schedule, type, path, suffix, channelData, g, k, iRate)

    ps.close()
    ps.join()
    print(u'主进程结束!')

import multiprocessing
import readAndWriteDataSet
import tools
import kmeans
import pca
import time
import numpy as np


def getCentroids(schedule, path, suffix, channelData, g, k, iRate, type=u'C'):
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

    schedule[1] += 1
    tmpSchedule = schedule[1]
    print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分开始！')

    # 得到相关系数矩阵并输出,然后放到一个矩阵中
    covMatrixList = tools.getCovMatrixList(channelData)
    informations, SigmaList, UList = tools.getInformations(covMatrixList)

    if type == u'total':
        # 对协方差进行聚类
        getCentroidsCore(path, suffix, channelData, covMatrixList, informations, SigmaList, UList, g, k, iRate, "C")
        # 对变换矩阵进行聚类
        getCentroidsCore(path, suffix, channelData, covMatrixList, informations, SigmaList, UList, g, k, iRate, "U")
    else:
        getCentroidsCore(path, suffix, channelData, covMatrixList, informations, SigmaList, UList, g, k, iRate, type)

    # 显示进度
    print(u'共' + str(schedule[0]) + u'部分，' + u'第' + str(tmpSchedule) + u'部分完成，' + u'已完成' + str(
        schedule[1]) + u'部分，' + u'完成度：' + '%.2f%%' % (
                  schedule[1] / schedule[0] * 100) + u'！')


def getCentroidsCore(path, suffix, channelData, covMatrixList, informations, SigmaList, UList, g, k, iRate, type="C"):
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    pathSuffix = type + "_" + str(g) + "_" + nowTime

    outOldCovMatrixListPath = path + "getCentroids_outOldCovMatrixList_" + pathSuffix
    outCentroidListPath = path + "getCentroids_outCentroidList_" + pathSuffix
    outClusterAssmentPath = path + "getCentroids_outClusterAssment_" + pathSuffix
    outNewChannelDataPath = path + "getCentroids_outNewChannelData_" + pathSuffix
    outNewCovMatrixListPath = path + "getCentroids_outNewCovMatrixList_" + pathSuffix
    ratesPath = path + "getCentroids_rates_" + pathSuffix
    UTsPath = path + "getCentroids_UTs_" + pathSuffix

    clusterAssmentList = []
    newChannelData = []
    newCovMatrixList = []
    UTs = []
    rates = []

    centroidList = []

    if type == u'C':
        allCovMatrix = tools.matrixListToMatrix(covMatrixList)

        # 对协方差进行聚类
        centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, k)
        clusterAssmentList.append(clusterAssment)
        centroidList = tools.matrixToMatrixList(centroids)

        # 分析PCA效果
        # newChannelData, newCovMatrixList, UTs, rates = pca.pca(channelData, informations, centroidList, clusterAssment, iRate)

    if type == u'U':
        allU = tools.matrixListToMatrix_U(UList)
        weights = tools.matrixListToMatrix_U(SigmaList)

        # 对协方差进行聚类
        centroids, clusterAssment = kmeans.KMeansOushi_U(allU, k, weights, iRate)
        clusterAssmentList.append(clusterAssment)
        centroidList = tools.matrixToMatrixList_U(centroids)

        # 分析PCA效果
        # newChannelData, newCovMatrixList, UTs, rates = pca.pca_U(channelData, informations, centroidList, clusterAssment, iRate)

    # 输出结果
    # 输出聚类结果
    # readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath, suffix)
    # 协方差矩阵太大了，先不输出
    # readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath, suffix)
    # 聚类中心太大了，先不输出
    readAndWriteDataSet.write(centroidList, outCentroidListPath, suffix)
    # 输出PCA结果
    # readAndWriteDataSet.write(newChannelData, outNewChannelDataPath, suffix)
    # readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixListPath, suffix)
    # readAndWriteDataSet.write(UTs, UTsPath, suffix)
    # readAndWriteDataSet.write(rates, ratesPath, suffix)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    schedule = manager.Array('i', [1, 0])

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
    iRate = 8  # 降维后维度
    k = 7  # 聚类中心数量
    type = u'total'

    for g in range(1, (1 << a) + 1):
        channelData = []
        for i in range(p):
            channelDataPage = channelDataAll[i]
            channelData.append(channelDataPage[:, (g - 1) * sub:g * sub])
        # ps.apply_async(getCentroids, args=(schedule, path, suffix, channelData, g, k, iRate, type))
        getCentroids(schedule, path, suffix, channelData, g, k, iRate, type)

    ps.close()
    ps.join()
    print(u'主进程结束!')

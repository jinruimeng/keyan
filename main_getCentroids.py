import readAndWriteDataSet
import getCovMatrix
import kmeans
import pca
import elbow
import time

if __name__ == '__main__':
    type = "mashi"
    # 读取数据
    channelDataPath = u'E:\\workspace\\keyan\\channelDate.xlsx'
    channelData = readAndWriteDataSet.excelToMatrix(channelDataPath)
    # 得到协方差的集合，并把协方差放到一个矩阵中
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    # 利用肘部法选择聚类中心个数
    # elbow.elbow(allCovMatrix, 1, 6, type)

    # 对协方差进行聚类
    k = 2
    if "mashi" == type:
        centroids, clusterAssment = kmeans.KMeansMashi(allCovMatrix, k)
    else:
        centroids, clusterAssment = kmeans.KMeansOushi(allCovMatrix, k)

    # 分析PCA效果
    centroidList = getCovMatrix.matrixToMatrixList(centroids)
    newChannelData, newCovMatrixList = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, 0.8)
    nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    outOldCovMatrixListPath = "E:\\workspace\\keyan\\getCentroids_outOldCovMatrixList_" + type + str(
        nowTime) + ".xlsx"
    outCentroidListPath = "E:\\workspace\\keyan\\getCentroids_outCentroidList_" + type + str(nowTime) + ".xlsx"
    outClusterAssmentPath = "E:\\workspace\\keyan\\getCentroids_outClusterAssment_" + type + str(nowTime) + ".xlsx"
    outNewChannelDataPath = "E:\\workspace\\keyan\\getCentroids_outNewChannelData_" + type + str(nowTime) + ".xlsx"
    outNewCovMatrixListPath = "E:\\workspace\\keyan\\getCentroids_outNewCovMatrixList_" + type + str(nowTime) + ".xlsx"

    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)

    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath)
    readAndWriteDataSet.write(centroidList, outCentroidListPath)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath)
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath)
    readAndWriteDataSet.write(newCovMatrixList, outNewCovMatrixListPath)

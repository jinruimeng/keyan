import readAndWriteDataSet
import kmeans
import getCovMatrix
import pca
import time

if __name__ == '__main__':
    type = "oushi"
    channelDataPath = u'E:\\workspace\\keyan\\channelDate.xlsx'
    centroidListPath = u'E:\\workspace\\keyan\\getCentroids_outCentroidList.xlsx'

    centroidList = readAndWriteDataSet.excelToMatrix(centroidListPath)
    channelData = readAndWriteDataSet.excelToMatrix(channelDataPath)

    # 计算出信道数据的协方差
    centroids = getCovMatrix.matrixListToMatrix(centroidList)
    covMatrixList = getCovMatrix.getCovMatrixList(channelData)
    allCovMatrix = getCovMatrix.matrixListToMatrix(covMatrixList)

    # 确定每个数据分别属于哪个簇
    clusterAssment = kmeans.getClusterAssment(allCovMatrix, centroids, type)

    # 分析PCA效果
    newChannelData, newCovMatrixs = pca.pca(channelData, covMatrixList, centroidList, clusterAssment, 0.8)

    outOldCovMatrixListPath = "E:\\workspace\\keyan\\cluster_outOldCovMatrixList" + str(time.time()) + ".xlsx"
    outClusterAssmentPath = "E:\\workspace\\keyan\\cluster_outClusterAssment" + str(time.time()) + ".xlsx"
    outNewChannelDataPath = "E:\\workspace\\keyan\\cluster_outNewChannelData" + str(time.time()) + ".xlsx"
    outNewCovMatrixsPath = "E:\\workspace\\keyan\\cluster_outNewCovMatrixList" + str(time.time()) + ".xlsx"
    clusterAssmentList = []
    clusterAssmentList.append(clusterAssment)

    readAndWriteDataSet.write(covMatrixList, outOldCovMatrixListPath)
    readAndWriteDataSet.write(clusterAssmentList, outClusterAssmentPath)
    readAndWriteDataSet.write(newChannelData, outNewChannelDataPath)
    readAndWriteDataSet.write(newCovMatrixs, outNewCovMatrixsPath)

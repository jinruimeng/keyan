import random
import numpy as np
import getDistance
import readAndWriteDataSet


# 从数据中随机选择k个作为起始的中心点
# 数据维度为n，数量为m
def randCent(dataSet, k):
    m, n = dataSet.shape  # 查看dataSet的维数
    centroids = np.array(np.zeros((k, n)), dtype=complex)
    rs = random.sample(range(0, m), k)  # 在0~m范围内生成k个随机数[0,m)
    for i in range(len(rs)):
        centroids[i, :] = dataSet[rs[i], :]
    return centroids


# k均值聚类
# 根据欧式距离聚类
def KMeansOushi(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.ones((m, 2)), dtype=complex)
    clusterChange = True
    count = 0

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        count += 1
        print("第" + str(count) + "次寻找聚类中心！")
        count2 = 0
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = getDistance.oushiDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                count2 += 1
            clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            if np.shape(pointsInCluster)[0] > 0:
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        for i in range(m):
            clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0].real), :],
                                                             dataSet[i, :])
        if count2 > 0:
            clusterChange = True

    print("Congratulations,cluster complete!")
    # print("centroids:\n", centroids)
    # print("clusterAssment:\n", clusterAssment)
    return centroids, clusterAssment


# k均值聚类
# 根据加权距离聚类
def KMeansOushi_U(dataSet, k, weights, newDimension):
    m, n = np.shape(dataSet)  # 行的数目
    p = (int)(n ** 0.5)
    if newDimension > p:
        newDimension = p
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.ones((m, 2)), dtype=complex)
    clusterChange = True
    count = 0

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        count += 1
        print("第" + str(count) + "次寻找聚类中心！")
        count2 = 0
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = -1
            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的加权距离
                distance = 0
                for a in range(p):
                    distance + getDistance.oushiDistance(centroids[j, a * p:a * p + newDimension],
                                                         dataSet[i, a * p:a * p + newDimension]) * weights[i, a]
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                count2 += 1
            clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            if np.shape(pointsInCluster)[0] > 0:
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        for i in range(m):
            distance = 0
            for j in range(newDimension):
                distance += getDistance.oushiDistance(
                    centroids[int(clusterAssment[i, 0].real), j * p:j * p + newDimension],
                    dataSet[i, j * p:j * p + newDimension]) * weights[i, j]
            clusterAssment[i, 1] = distance
        if count2 > 0:
            clusterChange = True

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


# 根据马氏距离聚类
def KMeansMashi(dataSet, k):
    # 备份原数据
    dataSetCopy = dataSet
    m = np.shape(dataSetCopy)[0]  # 行的数目
    n = np.shape(dataSetCopy)[1]  # 数据维度
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.ones((m, 2)))
    clusterChange = True
    # 存储每个簇中的数据
    dataSetOfCluster = []
    dataSetOfCluster.append(dataSetCopy)
    changeMatrixs = []

    # 第1步 初始化centroids
    centroids = randCent(dataSetCopy, k)
    while clusterChange:
        clusterChange = False

        # 计算变换矩阵
        covMatrix = np.array(np.zeros((n, n)))
        for i in range(len(dataSetOfCluster)):
            tmpDataSet = dataSetOfCluster[i]
            # 每一列代表一个维度
            covMatrix += np.cov(tmpDataSet, rowvar=False) * np.shape(tmpDataSet)[0]
        covMatrix = covMatrix / m
        U, Sigma, VT = np.linalg.svd(covMatrix)
        Sigma = np.diag(Sigma)
        changeMatrix = U * (Sigma ** 0.5) * VT
        changeMatrixs.append(changeMatrix)

        dataSetCopy = np.dot(dataSetCopy, changeMatrix)
        centroids = np.dot(centroids, changeMatrix)

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = getDistance.oushiDistance(centroids[j, :], dataSetCopy[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        dataSetOfCluster.clear()
        for i in range(k):
            pointsInCluster = dataSetCopy[np.nonzero(clusterAssment[:, 0].A == i)[0]]  # 获取簇类所有的点
            centroids[i, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
            dataSetOfCluster.append(pointsInCluster)
        for i in range(m):
            clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0]), :], dataSetCopy[i, :])

    for i in range(len(changeMatrixs)):
        IchangeMatrix = changeMatrixs[len(changeMatrixs) - i - 1]
        for j in range(n):
            if IchangeMatrix[j, j] != 0:
                IchangeMatrix[j, j] = IchangeMatrix[j, j] ** -1
            else:
                IchangeMatrix[j, j] = IchangeMatrix[j, j]
        centroids = np.dot(centroids, IchangeMatrix)

    # 以下三行是计算马氏距离所需
    allData = np.concatenate((dataSet, centroids))
    covMatrix = np.cov(allData, rowvar=False)
    IcovMatrix = np.linalg.inv(covMatrix)

    # outMatrixList = []
    # outMatrixList.append(allData)
    # outMatrixList.append(covMatrix)
    # outMatrixList.append(IcovMatrix)
    # path = "/Users/jinruimeng/Downloads/keyan/"
    # # path = "E:\\workspace\\keyan\\"
    # nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    # outSomeMatrixListPath = path + "getCentroids_outSomeMatrixList_mashi" + str(
    #     nowTime) + ".xlsx"
    # readAndWriteDataSet.write(outMatrixList, outSomeMatrixListPath)

    for i in range(m):
        clusterAssment[i, 1] = getDistance.mashiDistance(centroids[int(clusterAssment[i, 0]), :], dataSet[i, :],
                                                         IcovMatrix)
    print("Congratulations,cluster complete!")
    print("centroids:\n", centroids)
    print("clusterAssment:\n", clusterAssment)
    return centroids, clusterAssment


# 已知聚类中心，求样本属于哪个类
def getClusterAssment(dataSet, centroids, type="oushi"):
    m = np.shape(dataSet)[0]  # 行的数目
    k = np.shape(centroids)[0]  # 中心点的数量
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.array(np.ones((m, 2)), dtype=complex)

    # outMatrixList = []
    # outMatrixList.append(allData)
    # outMatrixList.append(covMatrix)
    # outMatrixList.append(IcovMatrix)
    # path = "/Users/jinruimeng/Downloads/keyan/"
    # # path = "E:\\workspace\\keyan\\"
    # nowTime = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime(time.time()))
    # outSomeMatrixListPath = path + "cluster_outSomeMatrixList_" + type + str(
    #     nowTime) + ".xlsx"
    # readAndWriteDataSet.write(outMatrixList, outSomeMatrixListPath)

    # 遍历所有的样本（行数）
    for i in range(m):
        minDist = 100000000.0
        minIndex = -1

        # 遍历所有的中心点
        # 第2步 找出最近的中心点
        for j in range(k):
            # 计算该样本到质心的距离
            if "mashi" == type:
                # 以下三行是计算马氏距离所需
                allData = np.concatenate((dataSet, centroids))
                covMatrix = np.cov(allData, rowvar=False)
                IcovMatrix = np.linalg.inv(covMatrix)
                distance = getDistance.mashiDistance(centroids[j, :], dataSet[i, :], IcovMatrix)
            else:
                distance = getDistance.oushiDistance(centroids[j, :], dataSet[i, :])
            if distance < minDist:
                minDist = distance
                minIndex = j

        # 第 3 步：更新每一行样本所属的簇
        clusterAssment[i, :] = minIndex, minDist
    return clusterAssment


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    k = 1
    centroids1, clusterAssment1 = KMeansOushi(dataSet, k)
    centroids2, clusterAssment2 = KMeansMashi(dataSet, k)

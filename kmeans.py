import random
import numpy as np
import readAndWriteDataSet
import getDistance


# 从数据中随机选择k个作为起始的中心点
# 数据维度为n，数量为m
def randCent(dataSet, k):
    m, n = dataSet.shape  # 查看dataSet的维数
    centroids = np.zeros((k, n))
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
    clusterAssment = np.mat(np.ones((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
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
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        for i in range(m):
            clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0]), :], dataSet[i, :])

    print("Congratulations,cluster complete!")
    print("centroids:\n", centroids)
    print("clusterAssment:\n", clusterAssment)
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
        for j in range(k):
            pointsInCluster = dataSetCopy[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
            dataSetOfCluster.append(pointsInCluster)
        for i in range(m):
            clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0]), :], dataSetCopy[i, :])

    for i in range(len(changeMatrixs)):
        centroids = np.dot(centroids, np.linalg.inv(changeMatrixs[len(changeMatrixs) - i - 1]))
    for i in range(m):
        clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0]), :], dataSet[i, :])
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
    clusterAssment = np.mat(np.ones((m, 2)))

    # 遍历所有的样本（行数）
    for i in range(m):
        minDist = 100000000.0
        minIndex = -1

        # 遍历所有的中心点
        # 第2步 找出最近的中心点
        for j in range(k):
            # 计算该样本到质心的欧式距离
            if "oushi" == type:
                distance = getDistance.oushiDistance(centroids[j, :], dataSet[i, :])
            else:
                distance = getDistance.mashiDistance(centroids[j, :], dataSet[i, :])
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

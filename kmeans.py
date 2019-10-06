import random
import numpy as np
import getDistance
import readAndWriteDataSet


def randCent(dataSet, centroidNum):
    m, n = dataSet.shape  # 查看dataSet的维数
    centroids = np.array(np.zeros((centroidNum, n)))
    rs = random.sample(range(0, m), centroidNum)  # 在0~m范围内生成k个随机数[0,m)
    for i in range(np.shape(rs)[0]):
        centroids[i, :] = dataSet[rs[i], :]
    return centroids


# 从数据中随机选择k个作为起始的中心点
def randCent2(dataSetList, centroidNum):
    p = np.shape(dataSetList)[0]  # 查看数据个数
    centroidList = []
    rs = random.sample(range(0, p), centroidNum)  # 在0~m范围内生成k个随机数[0,m)
    for i in range(np.shape(rs)[0]):
        centroidList.append(dataSetList[i])
    return centroidList


# k均值聚类
# 根据欧式距离聚类 dataSet为协方差矩阵
def KMeansOushi(dataSet, centroidNum):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.ones((m, 2)))
    clusterChange = True
    count = 0

    # 第1步 初始化centroids
    centroids = randCent(dataSet, centroidNum)
    while clusterChange:
        count += 1
        print("第" + str(count) + "次寻找聚类中心！")
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(centroidNum):
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
        for i in range(centroidNum):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0]]  # 获取簇类所有的点
            if np.shape(pointsInCluster)[0] > 0:
                centroids[i, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    for i in range(m):
        clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0].real), :], dataSet[i, :])

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


# 根据代价函数聚类 dataSet为信道信息矩阵
# k-聚类中心数量 npower-噪声功率 newDimension-保留维度
def KMeansOushi_costFunction(dataSet, dataSet2, covs, centroidNum, npower, iRate):
    m = np.shape(dataSet)[0]  # 样本数
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.zeros((m, 2)))
    clusterChange = True
    count = 0

    # 第1步 初始化centroids
    centroidList = randCent2(covs, centroidNum)
    UList = []
    for i in range(centroidNum):
        U, Sigma, VT = np.linalg.svd(centroidList[i])
        UList.append(np.transpose(VT)[:, 0:iRate])
    while clusterChange and count <= 10:
        count += 1
        print("第" + str(count) + "次寻找聚类中心！")
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = 1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(centroidNum):
                # 计算该样本到质心的代价函数
                distance = 1 - getDistance.costFunction(dataSet[i], dataSet2[i], UList[j], npower[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        for i in range(centroidNum):
            index = np.nonzero(clusterAssment[:, 0].A.real == i)[0]
            pointsInCluster = []
            for j in range(np.shape(index)[0]):
                pointsInCluster.append(covs[j])
            if np.shape(pointsInCluster)[0] > 0:
                tmpSum = pointsInCluster[0]
                for n in range(1, np.shape(pointsInCluster)[0]):
                    tmpSum += pointsInCluster[n]
                centroidList[i] = tmpSum / np.shape(pointsInCluster)[0]  # 求新的聚类中心

        UList = []
        for i in range(centroidNum):
            U, Sigma, VT = np.linalg.svd(centroidList[i])
            UList.append(np.transpose(VT)[:, 0:iRate])

    for i in range(m):
        clusterAssment[i, 1] = 1 - getDistance.costFunction(dataSet[i], dataSet2[i], UList[int(clusterAssment[i, 0].real)], npower[i])

    print("Congratulations,cluster complete!")
    return centroidList, UList, clusterAssment


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    k = 1

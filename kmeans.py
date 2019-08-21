import random
import numpy as np
import getDistance
import readAndWriteDataSet


# 从数据中随机选择k个作为起始的中心点
def randCent(dataSetList, k):
    p = np.shape(dataSetList)[0]  # 查看数据个数
    centroidList = []
    rs = random.sample(range(0, p), k)  # 在0~m范围内生成k个随机数[0,m)
    for i in range(len(rs)):
        centroidList.append(dataSetList[i])
    return centroidList


# k均值聚类
# 根据欧式距离聚类 dataSet为协方差矩阵
def KMeansOushi(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.ones((m, 2)))
    clusterChange = True
    count = 0

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
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
            if np.shape(pointsInCluster)[0] > 1:
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        for i in range(m):
            clusterAssment[i, 1] = getDistance.oushiDistance(centroids[int(clusterAssment[i, 0].real), :], dataSet[i, :])

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


# 根据代价函数聚类 dataSet为信道信息矩阵
# k-聚类中心数量 npower-噪声功率 newDimension-保留维度
def KMeansOushi_costFunction(dataSet, covs, k, npower, newDimension=5):
    m = np.shape(dataSet)[0]  # 样本数
    n = np.shape(dataSet)[2]  # 列数
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.matrix(np.zeros((m, 2)))
    clusterChange = True
    count = 0

    maxIs = []
    for i in range(m):
        maxI = getDistance.costFunction(dataSet[i], np.eye(n), npower[i])
        maxIs.append(maxI)

    # 第1步 初始化centroids
    centroidList = randCent(covs, k)
    UList = []
    for i in range(k):
        U, Sigma, VT = np.linalg.svd(centroidList[i])
        UList.append(np.transpose(U)[:, 0:newDimension])
    while clusterChange:
        count += 1
        print("第" + str(count) + "次寻找聚类中心！")
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000000.0
            minIndex = 1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的代价函数
                distance = maxIs[i] - getDistance.costFunction(dataSet[i], UList[j], npower[i])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist

        # 第 4 步：更新质心
        for i in range(k):
            index = np.nonzero(clusterAssment[:, 0].A.real == i)[0]
            pointsInCluster = []
            for j in range(len(index)):
                pointsInCluster.append(covs[j])
            if np.shape(pointsInCluster)[0] > 1:
                tmpSum = pointsInCluster[0]
                for n in range(1, np.shape(pointsInCluster)[0]):
                    tmpSum += pointsInCluster[n]
                centroidList[i] = tmpSum / np.shape(pointsInCluster)[0]  # 求新的聚类中心

        UList = []
        for i in range(k):
            U, Sigma, VT = np.linalg.svd(centroidList[i])
            UList.append(np.transpose(U)[:, 0:newDimension])

        for i in range(m):
            clusterAssment[i, 1] = maxIs[i] - getDistance.costFunction(dataSet[i], UList[int(clusterAssment[i, 0].real)], npower[i])

    print("Congratulations,cluster complete!")
    return centroidList, UList, clusterAssment


if __name__ == '__main__':
    datafile = u'E:\\workspace\\keyan\\test.xlsx'
    dataSet = np.array(readAndWriteDataSet.excelToMatrix(datafile))
    k = 1

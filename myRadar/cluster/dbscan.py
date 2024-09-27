from sklearn.cluster import DBSCAN
import numpy as np


def dbscan_cluster(posints, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(posints)


def dbscan_selectPoint(points, eps, ampSpec):
    """DBSCAN聚类并选择每个簇中幅值最大的点代表簇的坐标

    Args:
        points (M,N): 点云位置数组
        eps (float): 簇半径
        ampSpec (M,): 每个点云对应的幅值

    Returns:
        numpy.ndarray: (L,N)簇的坐标，L为簇的个数，N为点云的维度
    """
    if len(points) == 0:
        return []
    labels = dbscan_cluster(points, eps, 1)
    pointGroups = dict()
    for i in range(len(labels)):
        if pointGroups.get(labels[i]) is None:
            pointGroups[labels[i]] = [i]
        else:
            pointGroups[labels[i]].append(i)

    posCluster = []
    for key in pointGroups:
        ampMax = 0
        indexMax = None  # 保存簇里面幅值最大点在数组posPointsCloud中的索引
        for index in pointGroups[key]:
            if ampSpec[index] > ampMax:
                ampMax = ampSpec[index]
                indexMax = index
        posCluster.append(points[indexMax])
    return np.array(posCluster)

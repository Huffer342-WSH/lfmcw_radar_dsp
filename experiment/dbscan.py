# %%
""" 生成仿真数据 """
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go


centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.2)

X = StandardScaler().fit_transform(X)

go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers")], layout=go.Layout(title="原始数据")).show()

min_samples = 10
eps = 0.3
# %%
""" 使用scikit-learn中的DBSCAN算法 """

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# %%
""" 绘图 """


def draw(X, labels):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    figdata = []
    for k in unique_labels:

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        figdata.append(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker=dict(size=4, color=k),
                name=str(k),
            )
        )

        xy = X[class_member_mask & ~core_samples_mask]
        figdata.append(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker=dict(size=2, color=k),
                name=str(k),
            )
        )
    return go.Figure(data=figdata, layout=go.Layout(title=f"Estimated number of clusters: {n_clusters_}"))


# %%

draw(X, labels).show()


# %%


from collections import deque


def getneighborhoods(i, eps, D, neighbors):
    def idx(i, j):
        return i * (i - 1) // 2 + j

    count = 0
    for j in range(0, i):
        if D[idx(i, j)] <= eps:
            neighbors[count] = j
            count += 1
    for j in range(i + 1, N):
        if D[idx(j, i)] <= eps:
            neighbors[count] = j
            count += 1

    return neighbors[:count]


def dbscan_core(N, eps, min_samples, getneighborhoods, *args):
    _labels = np.full(N, -1)
    label_num = 0
    stack = deque()
    isVisited = set()
    for i in range(N):
        if _labels[i] != -1:
            continue
        neighbors = getneighborhoods(i, eps, *args)
        if i in isVisited:
            print(f"重复查询:{i}")
        isVisited.add(i)
        if len(neighbors) < min_samples:
            continue
        j = i
        _labels[j] = label_num
        while True:
            neighbors = getneighborhoods(j, eps, *args)
            if j in isVisited:
                print(f"重复查询:{j}")
            isVisited.add(j)
            if len(neighbors) >= min_samples:
                for v in neighbors:
                    if _labels[v] == -1:
                        stack.append(v)
                        _labels[v] = label_num
            if stack.__len__() == 0:
                break
            j = stack.pop()
        label_num += 1
    print(f"{isVisited.__len__()}个点被标记")
    return _labels


N = len(X)
D = np.zeros(shape=(N * (N - 1) // 2))
neighbors = np.zeros(shape=(N - 1), dtype=int)
for i in range(1, N):
    for j in range(0, i):
        D[i * (i - 1) // 2 + j] = np.linalg.norm(X[i] - X[j])
_labels = dbscan_core(N, eps, min_samples - 1, getneighborhoods, D, neighbors)

draw(X, _labels).show()

print(f" error:{np.sum(labels != _labels)}")

# %%

# %%
import sys

sys.path.append("../")

import numpy as np
from scipy.fft import fftshift, fft, fft2
import scipy.constants
import scipy.io
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field
import copy


from myRadar.arraysys import angleDualCh
from myRadar import polar2cart
from myRadar.cfar import cfar_2d, cfar_1d
from myRadar.lfmcw_radar_data_cube_generator import generateRadarDataCube
from myRadar.cluster import dbscan_selectPoint

import plotly.graph_objects as go
import drawhelp.draw as dh


# %% 加载数据

file = scipy.io.loadmat(file_name="../data/AT24G_RecordedData_丰海条形模块_单人行走.mat")


rdm = file["RDM"][1:].transpose(0, 1, 3, 2)
numFrame = file["numFrame"][0, 0] - 1
numChannel = file["numChannel"][0, 0]
numSample = file["numSample"][0, 0]
numRangeBin = file["numRangeBin"][0, 0]
numChirp = file["numChirp"][0, 0]
timeChirp = file["timeChirp"][0, 0]
timeChirpGap = file["timeChirpGap"][0, 0]
timeFrameGap = file["timeFrameGap"][0, 0]
timeFrame = (timeChirp + timeChirpGap) * numChirp + timeFrameGap
resRange = 0.71
resVelocity = scipy.constants.c / (24e9 * 2 * (timeChirp + timeChirpGap) * numChirp)

numTrain = (3, 4)
numGuard = (2, 4)
del file


@dataclass
class RadarPointCloud:
    """
    点云数据结构
    """

    idxR: int
    idxV: int
    radius: float  # 径向距离
    radialVelocity: float  # 径向速度
    theta: float  # 方位角
    amplitude: float  # 幅值
    snr: float  # 信噪比


def deletePoints(pointCloud, amp):
    bools = np.ones(shape=(len(pointCloud)), dtype=bool)
    a = (pointCloud[:-1, 0] == pointCloud[1:, 0]) & (pointCloud[:-1, 1] == pointCloud[1:, 1] - 1)
    b = amp[:-1] < amp[1:]
    bools[:-1] &= ~(a & b)
    bools[1:] &= ~(a & ~b)
    pointCloud = pointCloud[bools]
    return pointCloud


# %%
"""搜索点云"""

ampSpec2DList = np.abs(rdm)

# 遍历所有帧
indicesList_full = []
noiseList = []
snrList = []
indicesList = []
pointCloudList = []
for i in range(numFrame):
    amp = np.sum(ampSpec2DList[i], axis=0)
    _, noise = cfar_2d(amp, numTrain, numGuard, threshold=2, type="GOCA")
    snr = amp / noise
    potinsCloud = np.argwhere(np.logical_and(snr > 2.5, amp > 400))
    potinsCloud = potinsCloud[potinsCloud[:, 0] != 0]  # 去除零距离
    potinsCloud = potinsCloud[potinsCloud[:, 1] != 0]
    a = amp[potinsCloud[:, 0], potinsCloud[:, 1]]
    indicesList_full.append(potinsCloud)
    potinsCloud = deletePoints(potinsCloud, a)
    indicesList.append(potinsCloud)
    points = []
    for p in potinsCloud:
        points.append(
            RadarPointCloud(
                idxR=p[0],
                idxV=p[1],
                radius=p[0] * resRange,
                radialVelocity=((p[1] + numChirp / 2) % numChirp - numChirp / 2) * resVelocity,
                amplitude=amp[tuple(p)],
                snr=snr[tuple(p)],
                theta=angleDualCh(rdm[i, 0, p[0], p[1]], rdm[i, 1, p[0], p[1]]),
            )
        )
    pointCloudList.append(points)
    noiseList.append(noise)
    snrList.append(amp / noise)

# %%
"""  绘制幅度谱 """
dh.draw_2d_spectrumlist(ampSpec2DList[::1, 0, :, :], title="幅度谱").show()


# %%
""" 绘制信噪比 """
# dh.draw_2d_spectrumlist(snrList[::1], title="信噪比").show()

# %%
""" 绘制RDM的CFAR搜索结果 """
listData = []
for i in range(numFrame):
    data = list()
    data.append(go.Scatter(x=indicesList[i][:, 0], y=indicesList[i][:, 1], mode="markers", name="Raw"))
    listData.append(data)
fig = dh.draw_animation(listData, title="RDM的 GOCA-2DCFAR 搜索结果")
fig.update_layout(
    xaxis=dict(range=[-1, 17]),
    yaxis=dict(range=[-1, 33]),
    title="点云",
)
fig.show()

# %%
""" 绘制笛卡尔坐标系下的点云 """
listData = []
for i in range(numFrame):
    points = pointCloudList[i]
    x = [p.radius * np.cos(p.theta) for p in points]  #  if p.radialVelocity != 0
    y = [p.radius * np.sin(p.theta) for p in points]
    data = list()
    data.append(go.Scatter(x=x, y=y, mode="markers", name="Raw"))
    listData.append(data)
fig_pointClouds = dh.draw_animation(listData, title="点云-笛卡尔坐标系")
fig_pointClouds.update_layout(
    title="点云",
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
# %%
fig_pointClouds.show()


# %% 观察一帧数据
i = 217
go.Figure(data=[go.Surface(z=ampSpec2DList[i, 0])]).show()
go.Figure(data=[go.Surface(z=ampSpec2DList[i, 1])]).show()
go.Figure(data=[go.Surface(z=snrList[i])]).show()

# %%
for item in pointCloudList[i]:
    print(f"idx:[{item.idxR},{item.idxV}]", end=" ")
    print(
        f"theta:{item.theta / np.pi * 180}  [{np.angle(rdm[i, 0, item.idxR, item.idxV])/ np.pi * 180},{np.angle(rdm[i, 1, item.idxR, item.idxV])/ np.pi * 180}]",
        end=" ",
    )
    print(f"Mag:[{ampSpec2DList[i, 0][item.idxR, item.idxV]},{ampSpec2DList[i, 1][item.idxR, item.idxV]}]")
# %%

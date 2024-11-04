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


def loadmat(file_name):
    global rdm, numFrame, numChannel, numSample, numRangeBin, numChrip, timeChrip, timeChripGap, timeFrameGap, timeFrame, resRange, resVelocity
    file = scipy.io.loadmat(file_name=file_name)
    print(file["INFO"])
    rdm = file["RDM"][1:].transpose(0, 1, 3, 2)
    numFrame = file["numFrame"][0, 0] - 1
    numChannel = file["numChannel"][0, 0]
    numSample = file["numSample"][0, 0]
    numRangeBin = file["numRangeBin"][0, 0]
    numChrip = file["numChrip"][0, 0]
    timeChrip = file["timeChrip"][0, 0]
    timeChripGap = file["timeChripGap"][0, 0]
    timeFrameGap = file["timeFrameGap"][0, 0]
    timeFrame = (timeChrip + timeChripGap) * numChrip + timeFrameGap
    resRange = 0.71
    resVelocity = scipy.constants.c / (24e9 * 2 * (timeChrip + timeChripGap) * numChrip)
    del file


# loadmat("../data/AT24G_RecordedData_切向往复运动以反复遮挡远处角铁.mat")
loadmat("../data/AT24G_RecordedData_2m处静坐.mat")
# loadmat("../data/AT24G_RecordedData 2024-10-30 10-57-59.mat")
# loadmat("../data/AT24G_RecordedData 2024-11-04 16-31-21.mat")

numTrain = (3, 4)
numGuard = (2, 4)


@dataclass
class RadarPointCloud:
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
    potinsCloud = np.argwhere(np.logical_and(snr > 2.5, amp > 1500))
    potinsCloud = potinsCloud[potinsCloud[:, 0] != 0]  # 去除零距离
    # potinsCloud = potinsCloud[potinsCloud[:, 1] != 0]  # 去除零速度
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
                radialVelocity=((p[1] + numChrip / 2) % numChrip - numChrip / 2) * resVelocity,
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
dh.draw_2d_spectrumlist(snrList[::1], title="信噪比").show()

# %%
""" 提取角铁的信号 """
idxReflector = (3, 0)  # 角铁的信号在RDM中的坐标，观察RDM时可以看到
idxHuman = (2, 0)


signal2dfft_reflector = rdm[:, :, idxReflector[0], idxReflector[1]].transpose(1, 0)
amp_reflector = np.abs(signal2dfft_reflector)
phase_reflector = np.unwrap(np.angle(signal2dfft_reflector))

phaseDiff_reflector = (phase_reflector[0] - phase_reflector[1] + np.pi) % (2 * np.pi) - np.pi

signal2dfft_human = rdm[:, :, idxHuman[0], idxHuman[1]].transpose(1, 0)
amp_human = np.abs(signal2dfft_human)
phase_human = np.unwrap(np.angle(signal2dfft_human))
phaseDiff_human = (phase_human[0] - phase_human[1] + np.pi) % (2 * np.pi) - np.pi

# %% [markdown]
""" 绘图观察 

观察可见角铁所在单元的信号会在被遮挡和露出的瞬间发生突变，虽然幅度变化很大，但是相位变化小。

信号变化是角铁的露出和遮挡，但是角铁本身是静止的，复数值只是从人体信号的旁瓣切换到角铁信号，相位变化应该不会超过 $2\pi$

"""


go.Figure(
    data=[go.Scatter(y=(amp_reflector[0]), name="Rx0"), go.Scatter(y=(amp_reflector[1]), name="Rx1")],
    layout=go.Layout(title="角铁信号的幅度变化"),
).show()
go.Figure(
    data=[go.Scatter(y=(phase_reflector[0]), name="Rx0"), go.Scatter(y=(phase_reflector[1]), name="Rx1")],
    layout=go.Layout(title="角铁信号的相位变化"),
).show()

go.Figure(
    data=[
        go.Scatter3d(
            x=np.arange(len(signal2dfft_reflector[0])),
            y=np.real(signal2dfft_reflector[0]),
            z=np.imag(signal2dfft_reflector[0]),
            mode="lines+markers",
            marker=dict(size=3),
        ),
        go.Scatter3d(
            x=np.arange(len(signal2dfft_reflector[1])),
            y=np.real(signal2dfft_reflector[1]),
            z=np.imag(signal2dfft_reflector[1]),
            mode="lines+markers",
            marker=dict(size=3),
        ),
    ],
    layout=go.Layout(
        title="角铁信号复数域",
        scene=dict(
            aspectmode="manual", aspectratio=dict(x=5, y=1, z=1), xaxis=dict(title="Index"), yaxis=dict(title="Real Part"), zaxis=dict(title="Imaginary Part")
        ),
    ),
).show()
go.Figure(
    data=[go.Scatter(y=(phaseDiff_reflector))],
    layout=go.Layout(title="角铁信号两通道的相位差", yaxis_range=[-np.pi, np.pi]),
).show()


# %%
def getPhaseDiffAbs(phase, amp, windowSize):
    dd = np.diff(phase, axis=-1)
    dd[amp[1:] < 1000] = 0
    absdd = np.abs(dd)
    ret = np.array([np.sum(absdd[i : i + windowSize]) for i in range(len(absdd) - windowSize)])
    return ret


a = getPhaseDiffAbs(phase_reflector[0], amp_reflector[0], 40)
b = getPhaseDiffAbs(phase_human[0], amp_human[0], 40)

go.Figure(
    data=[go.Scatter(y=a, name="Rx0"), go.Scatter(y=b, name="Rx1")],
    layout=go.Layout(title="xx"),
).show()

# %%

# %%
import numpy as np
from scipy.fft import fftshift, fft, fft2
import scipy.constants
import scipy.io
from sklearn.cluster import DBSCAN


import joblib

from myRadar.arraysys import angleDualCh
from myRadar import polar2cart
from myRadar.cfar import cfar_2d, cfar_1d
from myRadar.lfmcw_radar_data_cube_generator import generateRadarDataCube
from myRadar.cluster import dbscan_selectPoint

import plotly.graph_objects as go
import drawhelp.draw as dh


# %% [markdown]
""" 
##  1. 设置雷达参数，并生成仿真数据

雷达参数包含四个部分：目标轨迹、波形、天线位置、采样配置

1. 目标轨迹用一个列表(list)储存，里面每一个元素是一个字典(dict)，每一个字典包含三个键值对
	- rsc：float 雷达调制周期
	- timestamps： (N,) 时间戳
	- positions：(N,3) 目标位置
    
2. 调制波形为LFMCW，除了基本的中心频率、调制带宽和调制时间外，还包含一些额外的参数：每个Frame包含的Chrip数，总Frame数，两个Chrip之间的间隔以及两个Frame之间的间隔，这些是实际的雷达中配置中需要的。
3. 天线位置包含发射天线的位置和接收天线的位置，每一个天线都用是一个(1,3)的数组，多个天线组成一个(N,3)的数组表示
4. 采样配置包含采样频率、以及每个调制周期的采样点数、
"""

# %%
# 配置目标轨迹
# targetsInfo包含多个目标的轨迹，每个目标用一个字典储存。
targetsInfo = []
trajTemp = scipy.io.loadmat("./data/mouse_trajectory_2024_09_13_17_22_54.mat")
targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel() * 10, pos=trajTemp["positions"] * 10))
trajTemp = scipy.io.loadmat("./data/mouse_trajectory_2024_09_13_17_23_04.mat")
targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel() * 10, pos=trajTemp["positions"] * 10))

# 配置波形
frequency = 24e9
bandwidth = 1000e6
timeChrip = 150e-6  # chirp调频的持续时间，和需要大于numSampling/freqSampling
timeIdle = 200e-6  # 每一个Chrip后的空闲时间，或者说是一帧中，两个chrip的间隔
timeNop = 2000e-6  # 一帧结束后的空闲时间
numChrip = 32

timeMin = 60
for i in targetsInfo:
    timeMin = min(timeMin, np.max(i["times"]))
numFrame = int(timeMin / ((timeChrip + timeIdle) * numChrip + timeNop))

# 配置天线
posTx = np.array([[0, 0, 0]])
posRx = np.array([[0, -0.25 * scipy.constants.c / frequency, 0], [0, 0.25 * scipy.constants.c / frequency, 0]])


# 配置采样
freqSampling = 1e6  # 采样频率
numSampling = 128

print(f"numFrame: {numFrame}")

# 生成数据
radarDataCube, posSeriesTargetsRaw = generateRadarDataCube(
    frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo
)

resRange = scipy.constants.c / (2 * bandwidth * (numSampling / freqSampling) / timeChrip)

# %% [markdown]
""" 
##  2. RDM分离目标后，使用相位差法测角
"""

# %%


def frameProcess_doa_phase(frames):
    """处理一帧数据，输出直角坐标系的检测结果

    Args:
        frames (np.ndarray): (2,numChrip,numFrame),一帧数据

    Returns:
        点云: (N,2)，第一列为X，第二列为Y
    """
    res = []
    if len(frames.shape) != 3:
        raise ValueError("frame must be 3D array")
    rdms = fftshift(fft2(frames, axes=(1, 2)), axes=1)
    specs = np.abs(rdms)
    numGuard = np.array([1, 1])
    numTrain = np.array([2, 4])
    (coords, _) = cfar_2d(specs[0], numTrain, numGuard, 3, type="Cross")
    for c in coords:
        r = c[1]
        theta = angleDualCh(rdms[0, c[0], c[1]], rdms[1, c[0], c[1]])
        res.append([r * np.cos(theta), r * np.sin(theta)])
    res = np.array(res)
    return res


posList = joblib.Parallel(n_jobs=-1)(joblib.delayed(frameProcess_doa_phase)(f) for f in radarDataCube)
# posList = [frameProcess_doa_phase(f) for f in radarDataCube]


# 绘图
listData = []
posSeriesTargets_FrameMean = np.mean(posSeriesTargetsRaw.reshape((2, -1, 4096, 3)), axis=2)
for i in range(len(posList)):
    coords = posList[i] * resRange
    if len(coords) == 0:
        listData.append([go.Scatter(x=posSeriesTargets_FrameMean[:, i, 0], y=posSeriesTargets_FrameMean[:, i, 1], mode="markers", name="Raw")])
    else:
        listData.append(
            [
                go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers", name="Detected"),
                go.Scatter(x=posSeriesTargets_FrameMean[:, i, 0], y=posSeriesTargets_FrameMean[:, i, 1], mode="markers", name="Raw"),
            ]
        )
fig = dh.draw_animation(listData[::10], title="目标检测结果——基于RDM+相位差法")
fig.update_layout(yaxis_range=[-9, 9], xaxis_range=[0, 18], xaxis_title="前后方向", yaxis_title="左右方向")
fig.show()

# dh.save_plotly_animation_as_video(fig, 30)

# %% [markdown]
""" ## 2. 取出一帧数据，对比相位差发在基于RDM和RangeFFT的情况下的差别

对于1T2R的雷达，只能从一组阵信号中得到一个目标的角度。为了雷达有更好的区分目标的能力，我们需要先在距离维度和速度维度上分离目标。

2DFFT得到的RDM分离了速度和距离维度，可以更好的区分目标，只做RangeFFT就只能分离速度维度。不能像在RDM的基础上测角一样可以分离同一距离单元内的不同速度的目标。

但是在低成本实际应用中，使用的MCU往往没有足够的内存和计算资源用于保存数据并计算Dopplor-FFT。所以可能需要使用基于RangeFFT的测角，对局部的距离单元可以保存数据做Dopplor-FFT再测角。

当然这只是对于1T2R的雷达来说，对于多通道的雷达来说，通过阵列信号处理是能够区分出多个目标的
"""

# %%================================ 准备一帧数据 ================================

# 获取两个目标位于同一个距离单元的帧的编号
posXDiff = np.abs(np.linalg.norm(posSeriesTargetsRaw[0], axis=1) - np.linalg.norm(posSeriesTargetsRaw[1], axis=1))  # 径向距离差
indexFrame_meet = np.unique(np.floor_divide(np.argwhere(posXDiff < 0.05), 4096))  # 两个目标径向距离相近的帧

indexFrame = indexFrame_meet[-1] + 20
print(f"Selected frame: {indexFrame}")
multiChannelFrame = radarDataCube[indexFrame]
posTargtes = np.mean(posSeriesTargetsRaw[:, (indexFrame - 1) * 4096 : indexFrame * 4096, :], axis=1)[:, :2]

# 绘制RDM
rdm = fftshift(fft2(multiChannelFrame), axes=1)
ampSpec2D = np.abs(rdm[0])
coords, noise_level = cfar_2d(ampSpec2D, (1, 3), (1, 1), 3)
dh.draw_spectrum(ampSpec2D / noise_level, title="RDM 信号幅值/噪声水平")

# %% ================================ RDM + 相位差法计算目标位置 ================================


def detectTarget_RDMAndPhaseDiff(frames, resRange, resVelocity, eps):
    r"""
    2DFFT+相位差法计算目标位置

    Parameters
    ----------
    frames : 3D array
        3维数组，shape=(numChannel, numChrip, numSampling)，雷达数据立方体
    resRange : float
        距离分辨率
    resVelocity : float
        速度分辨率
    eps : float
        DBSCAN聚类的最大距离

    Returns
    -------
    posCluster : 2D array
        2维数组，shape=(numCluster, 2)，目标的二维坐标
    """
    if frames.ndim != 3:
        raise ValueError("Input parameters 'frames' must be a 3D array")
    if frames.shape[0] <= 1:
        raise ValueError("Input parameters 'frames' must have at least 2 channel;")

    # 2DFFT
    rdm = fftshift(fft2(multiChannelFrame), axes=-2)

    # 2DFFT幅值谱
    ampSpec2D = np.sum(np.abs(rdm), axis=tuple(range(rdm.ndim - 2)))

    # CFAR检测
    idx2d, _ = cfar_2d(ampSpec2D, numTrain=(1, 3), numGuard=(1, 1), threshold=3)

    # 2DFFT幅值谱对应的距离和角度
    r = idx2d[:, 1] * resRange
    theta = np.array([angleDualCh(rdm[0, c[0], c[1]], rdm[1, c[0], c[1]]) for c in idx2d])

    # 将极坐标转换成二维坐标
    posPointsCloud = polar2cart(np.column_stack((r, theta)))

    # DBSCAN聚类
    posCluster = dbscan_selectPoint(posPointsCloud, eps=eps, ampSpec=ampSpec2D[idx2d[:, 0], idx2d[:, 1]])

    return posCluster


posCluster_RDM = detectTarget_RDMAndPhaseDiff(multiChannelFrame, resRange, 0, eps=0.9)
# ================================ RangeFFT + 相位差法计算目标位置 ================================

# 计算相位使用相干累加，计算位置使用非相干累加。
# 因为相干累加因为一帧的时间内通道间的目标造成相位差是几乎固定的，而噪声则是随机的，所以相干累加提高相位精度
# 但是对于幅度谱测距来说，快速目标的信号在多普勒维度（速度维度）相位变化较大，累加反而会抑制信号的幅度，因此测距应该选择非相干累加


def detectTarget_RangeFFTAndPhaseDiff(frames, resRange, resVelocity, eps):
    """
    RangeFFT + 相位差法目标检测

    Parameters
    ----------
    frames : 3D array
        3D array，shape=(numChannel, numChrip, numSampling), numChannel为通道数，numChrip为chrip数,numSampling为采样点数
    resRange : float
        距离分辨率
    resVelocity : float
        速度分辨率
    eps : float
        DBSCAN聚类的半径

    Returns
    -------
    posCluster : 2D array
        2维数组，shape=(numCluster, 2)，目标的二维坐标
    """
    if frames.ndim != 3:
        raise ValueError("Input parameters 'frames' must be a 3D array")
    if frames.shape[0] <= 1:
        raise ValueError("Input parameters 'frames' must have at least 2 channel;")
    RangeFFTFrame = fft(multiChannelFrame, axis=-1)
    ampSpec = np.sum(np.abs(RangeFFTFrame), axis=tuple(range(RangeFFTFrame.ndim - 1)))
    caSpec = np.sum(RangeFFTFrame, axis=-2)
    idx, noise_level = cfar_1d(ampSpec, numTrain=int(0.5 / resRange), numGuard=1, threshold=3)
    theta = angleDualCh(caSpec[0][idx], caSpec[1][idx])
    posPointsCloud = polar2cart(np.column_stack((idx * resRange, theta)))
    posCluster = dbscan_selectPoint(posPointsCloud, 3, ampSpec[idx])
    return posCluster


posCluster_RangeFFT = detectTarget_RangeFFTAndPhaseDiff(multiChannelFrame, resRange, 0, eps=0.9)

# ================================ 绘图比较 ================================
print(f"真实位置: {len(posTargtes)}\n簇的XY坐标为:\n {posTargtes}")

print(f"RDM + DBScan 聚类后得到的簇的个数: {len(posCluster_RDM)}\n簇的XY坐标为:\n {posCluster_RDM}")
print(f"RangeFFT + DBScan 聚类后得到的簇的个数: {len(posCluster_RangeFFT)}\n簇的XY坐标为:\n {posCluster_RangeFFT}")

# 把点绘制出来
go.Figure(
    data=[
        go.Scatter(x=posTargtes[:, 0], y=posTargtes[:, 1], mode="markers", name="真实位置"),
        go.Scatter(x=posCluster_RDM[:, 0], y=posCluster_RDM[:, 1], mode="markers", name="RDM"),
        go.Scatter(x=posCluster_RangeFFT[:, 0], y=posCluster_RangeFFT[:, 1], mode="markers", name="RangeFFT"),
    ],
    layout=go.Layout(
        title="聚类后的目标",
        yaxis_range=[-9, 9],
        xaxis_range=[0, 18],
        xaxis_title="前后方向",
        yaxis_title="左右方向",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    ),
).show()
# %%

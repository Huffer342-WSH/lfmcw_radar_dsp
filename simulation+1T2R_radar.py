# %%
import numpy as np
from scipy.fft import fftshift, fft, fft2
import scipy.constants
import scipy.io

import joblib

from myRadr.cfar import cfar_2d
from myRadr.lfmcw_radar_data_cube_generator import generateRadarDataCube

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
radarDataCube, targetsPosRaw = generateRadarDataCube(
    frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo
)

resRange = scipy.constants.c / (2 * bandwidth * (numSampling / freqSampling) / timeChrip)

# %% [markdown]
""" 
##  2. RDM分离目标后，使用相位差法测角
"""

# %%
"""RDM分离目标后，使用相位差法测角"""


def clac_theta(complex0, complex1):
    """
    使用相位差法计算角度

    Args:
        complex0: 第一个通道的复数信号
        complex1: 第二个通道的复数信号

    Returns:
        theta: 角度，弧度
    """

    phase0 = np.angle(complex0)
    phase1 = np.angle(complex1)
    phaseDelta = phase0 - phase1
    if phaseDelta < -np.pi:
        phaseDelta += 2 * np.pi
    elif phaseDelta > np.pi:
        phaseDelta -= 2 * np.pi
    theta = np.arcsin(phaseDelta / np.pi)
    return theta


def polar_to_cartesian(pos_polar):
    """
    将极坐标转换为直角坐标

    Args:
        pos_polar: 极坐标，形状为(N,2)的数组,第一列为极径，第二列为极角

    Returns:
        res: 直角坐标，形状为(N,2)的数组,第一列为X，第二列为Y
    """
    res = np.column_stack((pos_polar[:, 0] * np.cos(pos_polar[:, 1]), pos_polar[:, 0] * np.sin(pos_polar[:, 1])))
    return res


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
        theta = clac_theta(rdms[0, c[0], c[1]], rdms[1, c[0], c[1]])
        res.append([r * np.cos(theta), r * np.sin(theta)])
    res = np.array(res)
    return res


# posList = joblib.Parallel(n_jobs=-1)(joblib.delayed(frameProcess_doa_phase)(f) for f in radarDataCube)
posList = [frameProcess_doa_phase(f) for f in radarDataCube]


# 绘图
listData = []
posRaw = np.mean(targetsPosRaw.reshape((2, -1, 4096, 3)), axis=2)
for i in range(len(posList)):
    coords = posList[i] * resRange
    if len(coords) == 0:
        listData.append([go.Scatter(x=posRaw[:, i, 0], y=posRaw[:, i, 1], mode="markers", name="Raw")])
    else:
        listData.append(
            [
                go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers", name="Detected"),
                go.Scatter(x=posRaw[:, i, 0], y=posRaw[:, i, 1], mode="markers", name="Raw"),
            ]
        )
fig = dh.draw_animation(listData[::10], title="目标检测结果——基于RDM+相位差法")
fig.update_layout(yaxis_range=[-9, 9], xaxis_range=[0, 18], xaxis_title="前后方向", yaxis_title="左右方向")
fig.show()

dh.save_plotly_animation_as_video(fig, "simulation+1T2R_radar.mp4", 30)
# %%

# %%
import sys

sys.path.append("../")

import numpy as np
import scipy.constants
import scipy.io

from myRadar.tool.lfmcw_radar_data_cube_generator import generateRadarDataCube


# %%
# 配置目标轨迹
# targetsInfo包含多个目标的轨迹，每个目标用一个字典储存。
targetsInfo = []
trajTemp = scipy.io.loadmat("../data/mouse_trajectory_2024_09_13_17_22_54.mat")
targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel() * 10, pos=trajTemp["positions"] * 10))
trajTemp = scipy.io.loadmat("../data/mouse_trajectory_2024_09_13_17_23_04.mat")
targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel() * 10, pos=trajTemp["positions"] * 10))

numTargets = len(targetsInfo)

# 配置波形
frequency = 24e9
bandwidth = 250e6
timeChirp = 28e-06  # chirp调频的持续时间，和需要大于numSampling/freqSampling
timeFrame = 37e-3  # 一帧数据的持续时间
timeFrameGap = 12e-3  # 两帧之间的间隔
numChirp = 32
timeChirpGap = timeFrame / numChirp - timeChirp  # 两个chirp之间的时间间隔


timeMin = 60
for i in targetsInfo:
    timeMin = min(timeMin, np.max(i["times"]))
numFrame = int(timeMin / ((timeChirp + timeChirpGap) * numChirp + timeFrameGap))

# 配置天线
posTx = np.array([[0, 0, 0]])
posRx = np.array([[0, -0.25 * scipy.constants.c / frequency, 0], [0, 0.25 * scipy.constants.c / frequency, 0]])

# 配置采样
freqSampling = 1e6  # 采样频率
numSampling = 256

# 生成数据
print(f"numFrame: {numFrame}")

radarDataCube, posSeriesTargetsRaw = generateRadarDataCube(
    frequency, bandwidth, timeChirp, timeChirpGap, timeFrameGap, freqSampling, numSampling, numChirp, numFrame, posTx, posRx, targetsInfo
)
posSeriesTargets_FrameMean = np.mean(posSeriesTargetsRaw.reshape((posSeriesTargetsRaw.shape[0], -1, numChirp * numSampling, 3)), axis=2)
RDM = np.fft.fft2(radarDataCube, axes=(-1, -2))
# %% 保存数据
scipy.io.savemat(
    file_name="../data/RadarData_Simulate.mat",
    mdict=dict(
        tergatTrajectory=posSeriesTargets_FrameMean,
        radarDataCube=radarDataCube,
        RDM=RDM,
        timeChirp=timeChirp,
        timeChirpGap=timeChirpGap,
        timeFrame=timeFrame,
        timeFrameGap=timeFrameGap,
        timeFrameFull=timeFrame + timeFrameGap,
        frequency=frequency,
        bandwidth=bandwidth * (numSampling / freqSampling) / timeChirp,
        numSample=numSampling,
        numRangeBin=numSampling,
        numChirp=numChirp,
        numFrame=numFrame,
        numChannel=2,
    ),
    do_compression=True,
)

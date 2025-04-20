# %%
import numpy as np
import scipy.constants
import scipy.io
import typing
from dataclasses import dataclass, field
from scipy.fft import fftshift, fft, fft2
import copy
import drawhelp.draw as dh
import plotly.graph_objects as go

from myRadar.arraysys import angleDualCh
from myRadar.cfar import cfar_2d, cfar_1d


@dataclass
class RadarParam:
    """
    LFMCW雷达参数，包括雷达的波形以及采样的设置，主要是一些物理参数，一般开始运行后不会修改
    """

    frequency: float
    bandwidth: float
    timeChirp: float
    timeChirpGap: float
    timeFrameGap: float
    numPoint: int
    numRangeBin: int
    numChirp: int
    numChannel: int
    # 以下参数为衍生参数
    timeFrameVaild: float = field(init=False)
    resRange: float = field(init=False)
    resVelocity: float = field(init=False)

    def __post_init__(self):
        self.timeFrameVaild = (self.timeChirp + self.timeChirpGap) * self.numChirp
        self.resRange = scipy.constants.c / (2 * self.bandwidth)
        self.resVelocity = scipy.constants.c / (2 * self.frequency * self.timeFrameVaild)


@dataclass
class RadarConfig:
    """
    LFMCW雷达配置，包含CFAR，ROI等，主要是一些数据处理可配置的参数
    """

    numGuard: typing.Sequence
    numTrain: typing.Sequence
    cfarThreshold: float
    ampThreshold: float


@dataclass
class RadarPointCloud:
    """
    点云数据结构
    """

    radius: float  # 径向距离
    radialVelocity: float  # 径向速度
    theta: float  # 方位角
    amplitude: float  # 幅值
    snr: float  # 信噪比


class LFMCWRadarProcessor:
    # 常见的24G雷达参数，测试类用的
    commonParam24G = RadarParam(
        frequency=24e9,
        bandwidth=250e6,
        timeChirp=420e-6,
        timeChirpGap=1200e-6,
        timeFrameGap=3000e-6,
        numPoint=128,
        numRangeBin=25,
        numChirp=32,
        numChannel=2,
    )

    commonConfig = RadarConfig(
        numGuard=(1, 1),
        numTrain=(4, 5),
        cfarThreshold=3.5,
        ampThreshold=0.02,
    )

    def __init__(self, RadarParam: RadarParam, RadarConfig: RadarConfig, staticClutter: np.ndarray):
        # 复制雷达参数
        self.param = RadarParam
        self.config = RadarConfig

        self.pointClouds = []
        self.staticClutter = staticClutter  # 外部输入，会影响前几帧数据检测，运行时间久后会回到正常的值

        # 检查数据
        if staticClutter.shape != (self.param.numChannel, self.param.numRangeBin):
            raise ValueError(
                f"[LFMCWRadarProcessor]: 输入的静态噪声形状异常，期望的staticClutter.shape==({self.param.numChannel}, {self.param.numRangeBin}),"
                " 但实际输入的staticClutter.shape=={staticClutter.shape}"
            )

    def inputNewFrame(self, rdm: np.ndarray):
        # 0.检查输入的RDM的形状和数据类型
        # 由于在实际应用中，距离维度的FFT先计算，速度维度的FFT后计算，所以往往单通道RDM的第一个维度是距离，第二个维度是速度。
        # 此外，输入的RDM需要是没经过速度维度fftshift的，因为实际应用中，不会浪费时间去交换内存。
        if rdm.shape != (self.param.numChannel, self.param.numRangeBin, self.param.numChirp):
            raise ValueError(
                f"[LFMCWRadarProcessor]: 输入的RDM形状异常，期望的rdm.shape==({self.param.numChannel}, {self.param.numRangeBin}, {self.param.numChirp}),"
                " 分别对应{self.param.numChannel}个通道，{self.param.numRangeBin}个距离单元，{self.param.numChirp}个脉冲。\n 但实际输入的rdm.shape=={rdm.shape}"
            )
        if rdm.dtype != np.complex128 and rdm.dtype != np.complex64:
            raise ValueError(
                f"[LFMCWRadarProcessor]: 输入的RDM数据类型异常，期望的rdm.dtype==np.np.complex128或者np.np.complex64， 但实际输入的rdm.dtype=={rdm.dtype}"
            )

        ## 1. 信号处理
        """ 
        1. 信号处理
        处理雷达信号，得到点云数据
        """

        # 1.1 更新静态杂波。对于RDM来说，静态杂波约等于其速度维度上索引为0的数据在时间维度上的均值
        self.updateStaticClutter(rdm[:, :, 0])

        # 1.2 RDM减去静态杂波。
        rdm[:, :, 0] -= self.staticClutter

        # 计算幅度谱并cfar查找目标
        self.pointClouds = self.searchPeak_in_AmpSpec(
            rdm, self.config.numTrain, self.config.numGuard, thCFAR=self.config.cfarThreshold, thAMP=self.config.ampThreshold
        )

        # 在RDM索引层面第一次聚类点云，主要针对速度维频谱弥散的问题

        # 对点云超分辨率计算距离

        """
        2. 数据处理
        主要是目标跟踪，在点云数据的基础上，经过聚类、关联等数据处理得目标轨迹
        """

    def searchPeak_in_AmpSpec(self, rdm, numTrain, numGuard, thCFAR=1.5, thAMP=0.1, type="GOCA"):
        ampSpec2D = np.abs(rdm[0]) + np.abs(rdm[1])
        (_, noiselevel) = cfar_2d(ampSpec2D, numTrain, numGuard, thCFAR, type)
        snr = ampSpec2D / noiselevel

        # 从幅度谱中查找目标，同时满足信噪比和幅度谱两个条件。indices是有序的，第一个维度优先级高，第二个维度优先级低。C语言实现时也要满足
        indices = np.argwhere(np.logical_and(snr > thCFAR, ampSpec2D > thAMP))
        # print(indices)

        # 在同一个距离单元中，将连续的一个检测出来的点合并成一个从而减少速度弥散带来的影响
        bools = np.ones(shape=(len(indices)), dtype=bool)
        amp = ampSpec2D[indices[:, 0], indices[:, 1]]

        a = (indices[:-1, 0] == indices[1:, 0]) & (indices[:-1, 1] == indices[1:, 1] - 1)
        b = amp[:-1] < amp[1:]
        bools[:-1] &= ~(a & b)
        bools[1:] &= ~(a & ~b)
        indices = indices[bools]

        points = []
        for index in indices:
            point = RadarPointCloud(
                radius=index[0] * self.param.resRange,
                radialVelocity=(index[1] - self.param.numChirp / 2) * self.param.resVelocity,
                amplitude=ampSpec2D[tuple(index)],
                theta=angleDualCh(rdm[0, index[0], index[1]], rdm[1, index[0], index[1]]),
            )
            points.append(point)

        # dh.draw_spectrum(ampSpec2D)
        # snr[ampSpec2D < thAMP] = 0
        # dh.draw_spectrum(snr)
        # print(f"平均幅度:{np.mean(ampSpec2D)}")
        return points

    def updateStaticClutter(self, ChirpMean, weight=0.5):
        self.staticClutter = ChirpMean * (1 - weight) + self.staticClutter * weight

    def getPointsPosition(self):
        return [[p.radius * np.cos(p.theta), p.radius * np.sin(p.theta)] for p in self.pointClouds]

    def getObservation(self):
        return [[p.radius, p.theta, p.radialVelocity] for p in self.pointClouds]


# %%
# 导入数据

ENABLE_CPROFILE = False


mat = scipy.io.loadmat(
    file_name="./data/RadarData_Simulate.mat",
)
# 提取信号，添加直流偏置和噪声
radarDataCube = mat["radarDataCube"]
# radarDataCube += 1e-4 * (np.random.randn(*radarDataCube.shape) + 1j * np.random.randn(*radarDataCube.shape))
# radarDataCube += np.random.randn(radarDataCube.shape[-1]) + 1j * np.random.randn(radarDataCube.shape[-1])
radarParam = RadarParam(
    frequency=mat["frequency"][0, 0],
    bandwidth=mat["bandwidth"][0, 0],
    timeChirp=mat["timeChirp"][0, 0],
    timeChirpGap=mat["timeChirpGap"][0, 0],
    timeFrameGap=mat["timeFrameGap"][0, 0],
    numPoint=mat["numPoint"][0, 0],
    numRangeBin=35,
    numChirp=mat["numChirp"][0, 0],
    numChannel=mat["numChannel"][0, 0],
)
referPositionList = mat["tergatTrajectory"][:, :, :2].transpose(1, 0, 2)

chirpMean = fft2(radarDataCube[0], axes=(-2, -1))[:, 0, :]

# %%
# 测试
if ENABLE_CPROFILE:
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

# 创建对象
pointsPositionList = []
core = LFMCWRadarProcessor(RadarParam=radarParam, RadarConfig=LFMCWRadarProcessor.commonConfig, staticClutter=chirpMean[:, : radarParam.numRangeBin])

# 测试类
# rdm = fftshift(fft2(radarDataCube[200], axes=(-2, -1)), axes=-2)[:, :, :128]
# core.inputNewFrame(rdm)
# pointsPosition = core.getPointsPosition()
# print(pointsPosition)

indexFrame = list(range(0, 2000, 1))
for i in indexFrame:
    rangeFFT = fft(radarDataCube[i], axis=-1)[:, :, : radarParam.numRangeBin]
    rdm = fft(rangeFFT, axis=-2)
    core.inputNewFrame(rdm.transpose(0, 2, 1))
    pointsPosition = core.getPointsPosition()
    pointsPositionList.append(pointsPosition)

if ENABLE_CPROFILE:
    pr.disable()
    pr.dump_stats("output.prof")

# %% 绘图
listData = []
for i in range(0, len(indexFrame), 20):
    data = list()
    referPosition = referPositionList[indexFrame[i]]
    data.append(go.Scatter(x=referPosition[:, 0], y=referPosition[:, 1], mode="markers", name="Raw"))
    if len(pointsPositionList[i]) != 0:
        pointsPosition = np.array(pointsPositionList[i])
        data.append(go.Scatter(x=pointsPosition[:, 0], y=pointsPosition[:, 1], mode="markers", name="Detected"))

    listData.append(data)
fig = dh.draw_animation(listData, title="目标检测结果——基于RDM+相位差法")
fig.update_layout(
    xaxis=dict(range=[0, 18], scaleanchor="y", scaleratio=1),  # 设置x轴范围和缩放比例
    yaxis=dict(range=[-9, 9], scaleanchor="x", scaleratio=1),  # 设置y轴范围和缩放比例
    title="点云",  # 添加标题
)
fig.show()
# %%
# dh.save_plotly_animation_as_video(fig, fps=20)


# %%
def showOneFrame(index):
    frame = radarDataCube[index, 0]
    frame = frame.transpose(1, 0)
    rdm = fft2(frame, axes=(-2, -1))
    ampSepc = np.abs(rdm)
    coords, noise = cfar_2d(ampSepc, (1, 1), (4, 5), 3.5, type="GOCA")
    dh.draw_spectrum(ampSepc)
    dh.draw_spectrum(ampSepc / noise)
    print(coords)


showOneFrame(1500)

# %%

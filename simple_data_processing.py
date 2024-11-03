# %%
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

file = scipy.io.loadmat(file_name="./data/AT24G_RecordedData_运动人体_长方形轨迹.mat")


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
    potinsCloud = np.argwhere(np.logical_and(snr > 2.5, amp > 1500))
    potinsCloud = potinsCloud[potinsCloud[:, 0] != 0]  # 去除零距离
    potinsCloud = potinsCloud[potinsCloud[:, 1] != 0]  # 去除零速度
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
fig = dh.draw_animation(listData, title="点云-笛卡尔坐标系")
fig.update_layout(
    title="点云",
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
fig.show()


# %%
""" 点云聚类 """
from sklearn.cluster import DBSCAN

objectsList = []
res = np.array([1.5, 10, 0.52])
for i in range(numFrame):
    pc = pointCloudList[i]
    objects = []
    if len(pc) != 0:
        points = [[p.idxR / res[0], p.idxV / res[1], p.theta / res[2]] for p in pc]
        labels = DBSCAN(eps=1, min_samples=1).fit_predict(points)
        num = max(labels) + 1
        maxAmpList = np.zeros(num)
        maxIdxList = np.zeros(num, dtype=np.int32)
        for i in range(len(labels)):
            if labels[i] == -1:
                continue
            if maxAmpList[labels[i]] < pc[i].amplitude:
                maxAmpList[labels[i]] = pc[i].amplitude
                maxIdxList[labels[i]] = i
        for i in range(num):
            objects.append(pc[maxIdxList[i]])

    objectsList.append(objects)

# %%
""" 绘制聚类后平面点云 """
listData = []
for i in range(numFrame):
    points = objectsList[i]
    x = [p.radius * np.cos(p.theta) for p in points]  #  if p.radialVelocity != 0
    y = [p.radius * np.sin(p.theta) for p in points]
    data = list()
    data.append(go.Scatter(x=x, y=y, mode="markers", name="Raw"))
    listData.append(data)
fig = dh.draw_animation(listData, title="目标检测结果——基于RDM+相位差法")
fig.update_layout(
    title="点云",
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
fig.show()

# %%
""" 根据聚类后点云附近的频谱特性就行超分辨率测距 """
_objectsList = copy.deepcopy(objectsList)
for i in range(numFrame):
    amp = ampSpec2DList[i, 0]
    objects = []
    for j, point in enumerate(_objectsList[i]):
        idx0 = point.idxR
        idx1 = point.idxV
        a = idx0
        if idx0 + 1 < amp.shape[0] and idx1 - 1 > 0:
            if amp[idx0 + 1, idx1] > amp[idx0 - 1, idx1]:
                b = idx0 + 1
                r = point.amplitude / amp[idx0 + 1, idx1]
            else:
                b = idx0 - 1
                r = point.amplitude / amp[idx0 - 1, idx1]
        else:
            b = idx0
            r = 1
        x = (a * r + b) / (r + 1)
        _objectsList[i][j].idxR = x
        _objectsList[i][j].radius = x * resRange


# %%
""" 绘制聚类并超分辨率后的平面点云 """
listData = []
for i in range(10, numFrame):

    data = list()
    x = []
    y = []
    for j in range(i - 10, i):
        x.extend([p.radius * np.cos(p.theta) for p in _objectsList[j]])
        y.extend([p.radius * np.sin(p.theta) for p in _objectsList[j]])
    data.append(go.Scatter(x=x, y=y, mode="markers", name="超分辨率后"))
    listData.append(data)
    x = []
    y = []
    for j in range(i - 10, i):
        x.extend([p.radius * np.cos(p.theta) for p in objectsList[j]])
        y.extend([p.radius * np.sin(p.theta) for p in objectsList[j]])
    data.append(go.Scatter(x=x, y=y, mode="markers", name="原始点云"))
    listData.append(data)
fig = dh.draw_animation(listData, title="超分辨率后点云")
fig.update_layout(
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
fig.show()


# %%
"""  使用stonesoup的拓展卡尔曼+JPDA实现目标跟踪 """
from datetime import datetime, timedelta
from stonesoup.types.detection import Detection

start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

# 状态向量: [x,vx,y,vy]'
# 观测模型：[r,theta]'

# 设置观测模型
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


measurement_model = CartesianToBearingRange(  # 内置的 笛卡尔坐标系转极坐标系 观测模型
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 0.8]),  # Covariance matrix. 0.2 degree variance in
    translation_offset=np.array([[0], [0]]),  # 平移偏移，传感器位置
)

# 添加观测值
all_measurements = []
timesteps = [start_time]
for i, objs in enumerate(_objectsList):
    timestamp = start_time + timedelta(seconds=i * timeFrame)
    timesteps.append(timestamp)
    measurement_set = set()
    for p in objs:
        p: RadarPointCloud
        measurement = np.array([p.theta, p.radius])
        measurement_set.add(Detection(measurement, timestamp=timestamp, measurement_model=measurement_model))
    all_measurements.append(measurement_set)

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=timedelta(seconds=5) / (max(timesteps) - min(timesteps)))
plotter.plot_measurements(all_measurements, [0, 2])
plotter.fig


# 配置拓展卡尔曼滤波器

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])
predictor = ExtendedKalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater

updater = ExtendedKalmanUpdater(measurement_model)


# 配置JPDA

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA

hypothesiser = PDAHypothesiser(predictor=predictor, updater=updater, clutter_spatial_density=0.125, prob_detect=0.9)

data_associator = JPDA(hypothesiser=hypothesiser)


# 运行JPDA与拓展卡尔曼滤波

from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.functions import gm_reduce_single
from stonesoup.types.update import GaussianStateUpdate

prior1 = GaussianState([[0.43], [0], [-0.8], [0]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=timesteps[50])

tracks = {Track([prior1])}

for n in range(50, len(all_measurements)):
    measurements = all_measurements[n]
    hypotheses = data_associator.associate(tracks, measurements, timesteps[n])

    # Loop through each track, performing the association step with weights adjusted according to
    # JPDA.
    for track in tracks:
        track_hypotheses = hypotheses[track]

        posterior_states = []
        posterior_state_weights = []
        for hypothesis in track_hypotheses:
            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
            else:
                posterior_state = updater.update(hypothesis)
                posterior_states.append(posterior_state)
            posterior_state_weights.append(hypothesis.probability)

        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)

        # Reduce mixture of states to one posterior estimate Gaussian.
        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        # Add a Gaussian state approximation to the track.
        track.append(GaussianStateUpdate(post_mean, post_covar, track_hypotheses, track_hypotheses[0].measurement.timestamp))

plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
# %%
# 显示跟踪结果

plotter.fig.update_layout(
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
plotter.fig.show()

# %%
# dh.save_plotly_animation_as_video(plotter.fig, fps=20)

# %% 观察一帧数据
i = 181
go.Figure(data=[go.Surface(z=ampSpec2DList[i, 0])]).show()
go.Figure(data=[go.Surface(z=ampSpec2DList[i, 1])]).show()
go.Figure(data=[go.Surface(z=noiseList[i])]).show()
go.Figure(data=[go.Surface(z=snrList[i])]).show()
print(indicesList_full[i])
print(pointCloudList[i])
# %%

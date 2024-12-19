# %%
import sys

sys.path.append("../")


import numpy as np
from scipy.fft import fftshift, fft, fft2
import scipy.constants
import scipy.io
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

from sklearn.cluster import DBSCAN
from collections import deque

from myRadar.cfar import cfar_2d, cfar_result_filtering
from myRadar.base import BaseBasicData

from myRadar.plot import genScatterPolar
import myRadar.plot.draw as dh


import plotly.graph_objects as go

import datetime


# %%
# 跟踪器

# 自己实现的GNN目标跟踪
from scipy.optimize import linear_sum_assignment

import myRadar.track.kalman as mk
import myRadar.track.models as mm
from myRadar.base import Printable
from stonesoup.types.array import StateVector, StateVectors


class LifeCycle(Printable):
    def __init__(self, score):
        self.score = score
        self.unassociated_time = 0  #  关联失败的时间
        self.deducted_score = 0
        self.measurements = []
        self.post_measurements = []
        self.mese = 0


class TrackedTarget:
    next_uuid = 0

    def __init__(self, measurement, timestamp: datetime):
        TrackedTarget.next_uuid += 1

        self.uuid = TrackedTarget.next_uuid
        self.state = self.__clac_init_state(measurement, timestamp)
        self.life_cycle = LifeCycle(Initiator.initial_score)
        self.life_cycle.measurements.append(measurement)
        self.life_cycle.post_measurements.append(measurement)

    def __clac_init_state(self, measurement, timestamp):
        theta, phi, rho, rho_rate = measurement
        x = rho * np.cos(theta) * np.cos(phi)
        y = rho * np.cos(theta) * np.sin(phi)
        z = rho * np.sin(theta)
        vx = rho_rate * np.cos(theta) * np.cos(phi)
        vy = rho_rate * np.cos(theta) * np.sin(phi)
        vz = rho_rate * np.sin(theta)
        state = mk.GaussianState(state_vector=np.array([x, vx, y, vy, z, vz]).reshape(-1, 1), covar=np.diag([1, 0.5, 1, 0.5, 1, 0.5]) ** 2, timestamp=timestamp)
        return state


class Associator(Printable):
    def __init__(self, predictor: mk.KalmanPredictor, updater: mk.KalmanUpdater):
        self.predictor = predictor
        self.updater = updater

    def associate(self, targets, measurements, timestamp, missed_distance):

        prediction = [self.predictor.predict(i.state, timestamp) for i in targets]
        predicted_measurements = [self.updater.predict_measurement(i) for i in prediction]

        # 为每一个被跟踪的目标生成假设，为关联成功的目标的假设的measurement是None
        hypotheses = [
            mk.Hypothesis(prior_state=tar.state, prediction=ps, measurement=None, measurement_prediction=pm)
            for tar, ps, pm in zip(targets, prediction, predicted_measurements)
        ]
        row_map = []  # 距离矩阵的列下标 ---> tracked_targets 的下标
        distance_matrix = []  # 生成距离矩阵
        for i in range(len(predicted_measurements)):
            a = predicted_measurements[i].state_vector
            allinf_flag = True
            row = np.empty(len(measurements))
            for j, measurement in enumerate(measurements):
                b = measurement
                dis = np.linalg.norm((a - b).reshape(-1, 1) * (np.array([0, 2, 1, 1.5])))
                if dis < missed_distance:
                    allinf_flag = False
                else:
                    dis = np.inf
                row[j] = dis
            if not allinf_flag:
                distance_matrix.append(row)
                row_map.append(i)
        distance_matrix = np.array(distance_matrix)
        self_matrix = np.full((distance_matrix.shape[0], distance_matrix.shape[0]), np.inf)
        np.fill_diagonal(self_matrix, missed_distance)
        distance_matrix = np.column_stack((distance_matrix, self_matrix))
        print(f"距离矩阵:\n{distance_matrix}\n")

        # 最小和指派
        row4col, col4row = linear_sum_assignment(distance_matrix, 0)

        # 给每一个假设分配测量值
        for i, j in zip(row4col, col4row):
            if distance_matrix[i, j] != np.inf and j < len(measurements):
                hypotheses[row_map[i]].measurement = measurements[j]
                print(f"目标 {targets[row_map[i]].uuid}:\n {targets[row_map[i]].state.state_vector} \n  关联测量 {measurements[j]}")
        unassociated_measurements = [m for i, m in enumerate(measurements) if i not in col4row]

        # 按照假设更新每一个目标
        for target, hypothesis in zip(targets, hypotheses):
            if hypothesis.measurement is None:
                target.state = hypothesis.prediction
            else:
                target.state = self.updater.update(hypothesis)
        return hypotheses, unassociated_measurements


class Initiator(Printable):
    max_score = 3000
    initial_score = 1000

    def __init__(
        self,
        predictor: mk.KalmanPredictor,
        updater: mk.KalmanUpdater,
        associator: Associator,
        unassociated_time=2.0,  # 目标关联失败超时时间
        keep_motion_time=2.0,  # 目标是连续运动时，多少时间关联成功
        keep_static_time=8.0,  # 目标是静止时，多少时间关联成功
        speed_threshold=0.1,  # 速度阈值
        missed_distance=5.0,
    ):
        self.predictor = predictor
        self.updater = updater
        self.associator = associator

        score = Initiator.max_score - Initiator.initial_score
        self.unassociated_time = unassociated_time
        self.speed_threshold = speed_threshold
        self.unassociated_score = int(-Initiator.initial_score / unassociated_time)
        self.motion_score = int(score / keep_motion_time)
        self.static_score = int(score / keep_static_time)
        self.missed_distance = missed_distance

    def updateLifeCycle(self, life_cycle: LifeCycle, hypothesis: mk.Hypothesis, post: mk.GaussianState):
        score = 0
        dt = (hypothesis.prediction.timestamp - hypothesis.prior_state.timestamp).total_seconds()
        if hypothesis.measurement is None:
            # 关联失败
            if life_cycle.unassociated_time > self.unassociated_time:
                score -= int(life_cycle.score) // 2
            score += self.unassociated_score * dt
            life_cycle.unassociated_time += dt
            life_cycle.measurements.append(life_cycle.measurements[-1])
        else:
            score -= life_cycle.unassociated_time * self.unassociated_score / 2
            speed = abs(hypothesis.measurement[3])
            if speed > self.speed_threshold:
                score += self.motion_score * dt
            else:
                score += self.static_score * dt
            life_cycle.unassociated_time = 0
            life_cycle.measurements.append(hypothesis.measurement)
        life_cycle.score += int(score)
        life_cycle.post_measurements.append(self.updater.measurement_model.function(post.state_vector))
        if len(life_cycle.measurements) > int(3 / dt):
            del life_cycle.measurements[0]
            del life_cycle.post_measurements[0]

        # 计算测量值误差方差
        a = np.array(life_cycle.measurements)
        b = np.array(life_cycle.post_measurements)
        life_cycle.mese = (np.linalg.norm(a - b, axis=0) / len(a)).reshape(-1)
        if np.sum(life_cycle.mese) > 1:
            life_cycle.score = -1

    def initiate(self, confirmed_targets, unconfirmed_targets: list, measurements, timestamp):

        # 关联
        hypotheses, unassociated_measurements = self.associator.associate(unconfirmed_targets, measurements, timestamp, self.missed_distance)

        # 更新
        for target, hypothesis in zip(unconfirmed_targets, hypotheses):
            target: TrackedTarget
            hypothesis: mk.Hypothesis
            # 滤波
            if hypothesis.measurement is None:
                target.state = hypothesis.prediction
            else:
                target.state = self.updater.update(hypothesis)
            # 更新生命周期
            self.updateLifeCycle(target.life_cycle, hypothesis, target.state)

        # 删除无效目标，添加确认目标
        for t in reversed(unconfirmed_targets):
            if t.life_cycle.score < 0:
                unconfirmed_targets.remove(t)
                print(f"删除起始阶段目标 {t.uuid}")
            elif t.life_cycle.score > Initiator.max_score:
                print(f"添加到跟踪列表 {t.uuid}")
                confirmed_targets.append(t)
                unconfirmed_targets.remove(t)

        ### 仍然没有被关联的测量值，用于创建新目标
        for measurement in unassociated_measurements:
            theta, phi, rho, rho_rate = measurement
            # 速度慢的测量值不用于创建新目标
            if np.abs(rho_rate) < 0.1:
                continue
            unconfirmed_targets.append(TrackedTarget(measurement, timestamp=timestamp))


class Deleter(Printable):
    max_score = 5000

    def __init__(
        self,
        updater: mk.KalmanUpdater,
        unassociated_time=10.0,  # 目标关联失败超时时间
        missed_probability=0.2,  # 目标丢失的概率
    ):
        self.updater = updater

        self.unassociated_time = unassociated_time
        self.unassociated_score = int(-Deleter.max_score / unassociated_time)
        self.missed_probability = missed_probability

    def updateLifeCycle(self, life_cycle: LifeCycle, hypothesis: mk.Hypothesis, post: mk.GaussianState):
        score = 0
        dt = (hypothesis.prediction.timestamp - hypothesis.prior_state.timestamp).total_seconds()
        if hypothesis.measurement is None:
            # 关联失败
            life_cycle.unassociated_time += dt
            score += self.unassociated_score * dt
            life_cycle.measurements.append(life_cycle.measurements[-1])
        else:
            # 返回分数
            score += min(life_cycle.unassociated_time, dt / self.missed_probability) * self.unassociated_score
            score -= self.unassociated_score * dt
            life_cycle.unassociated_time = 0
            life_cycle.measurements.append(hypothesis.measurement)

        life_cycle.score += int(score)

        life_cycle.post_measurements.append(self.updater.measurement_model.function(post.state_vector))
        if len(life_cycle.measurements) > int(3 / dt):
            del life_cycle.measurements[0]
            del life_cycle.post_measurements[0]
            # 计算测量值误差方差

        a = np.array(life_cycle.measurements)
        b = np.array(life_cycle.post_measurements)
        life_cycle.mese = (np.linalg.norm(a - b, axis=0) / len(a)).reshape(-1)

        if np.sum(life_cycle.mese) > 1:
            life_cycle.score = life_cycle.score / 4

        angle = np.arctan2(post.state_vector[2], post.state_vector[0])
        r = np.linalg.norm([post.state_vector[0], post.state_vector[2]])
        if (angle > np.pi / 3) or (angle < -np.pi / 3):
            life_cycle.score = -1
        if r > 15:
            life_cycle.score = -1

        if life_cycle.score > Deleter.max_score:
            life_cycle.score = Deleter.max_score

    def delete(self, targets: list, hypotheses):
        for target, hypothesis in zip(targets, hypotheses):
            self.updateLifeCycle(target.life_cycle, hypothesis, target.state)
        for target in reversed(targets):
            if target.life_cycle.score < 0:
                targets.remove(target)


def track(associator, deleter, initiator, tracked_targets, unconfirmed_targets, measurements, missed_distance, timestamp):
    # 数据关联，得到假设和未关联的测量
    hypotheses, unassociated_measurements = associator.associate(tracked_targets, measurements, timestamp, missed_distance)

    # 删除无效目标
    deleter.delete(tracked_targets, hypotheses)

    # 航迹起始
    initiator.initiate(tracked_targets, unconfirmed_targets, unassociated_measurements, timestamp)


# 创建工作类  预测、更新、关联、起始
velocity_noise_coef = 1.5
meas_noise_covar = np.diag([0, 5 / 180 * np.pi, 0.2, 0.01]) ** 2

ekf_predictor = mk.KalmanPredictor(mm.TransitionModel(velocity_noise_coef))
ekf_updater = mk.KalmanUpdater(mm.MeasurementModel(meas_noise_covar))
associator = Associator(ekf_predictor, ekf_updater)
_deleter = Deleter(updater=ekf_updater, unassociated_time=10.0, missed_probability=0.2)
_initiator = Initiator(
    predictor=ekf_predictor,
    updater=ekf_updater,
    associator=associator,
    unassociated_time=2.0,
    keep_motion_time=2.0,
    keep_static_time=-10.0,
    speed_threshold=0.1,
    missed_distance=5,
)
tracked_targets = []  # 被跟踪的目标
unconfirmed_targets = []  # 航迹起始阶段的目标

# %%
# 处理器


class CFAR2dPoint(BaseBasicData):
    attributes = ["idx0", "idx1", "mag", "snr"]


class Measurement(BaseBasicData):
    attributes = ["azimuth", "distance", "velocity", "mag", "snr"]


@dataclass
class Radarparam:
    numChrip: int
    numRangeBin: int
    numChannel: int
    lambda_over_d: float
    resRange: float
    resVelocity: float


@dataclass
class RadarBasicData:
    mag: np.ndarray
    multi_frame_meas: deque
    measurements: list


@dataclass
class RadarCFARConfig:
    numTrain: np.ndarray
    numGuard: np.ndarray
    thSNR: float
    thMag: float


@dataclass
class RadarCFARFilterConfig:
    range0: int
    range1: int
    shape1: int
    th: float


@dataclass
class RadarConfig:
    cfar_cfg: RadarCFARConfig
    cfar_filter_cfg: RadarCFARFilterConfig
    channel_phase_diff_threshold: float


# %%


class Processor:
    def __init__(self, param: Radarparam, config: RadarConfig):
        self.param = param
        self.basic = RadarBasicData(
            mag=np.zeros(shape=(param.numRangeBin, param.numChrip)),
            multi_frame_meas=deque(maxlen=8),
            measurements=[],
        )
        self.config = config
        pass

    def __call__(self, rdms: np.ndarray, timestamp: datetime.datetime):

        # 幅度谱
        self.basic.mag = np.sum(np.abs(rdms), axis=0)

        # CFAR搜索点
        cfar_indices = self.__cfar2d_goca()

        # CFAR结果过滤
        cfar_indices = self.__cfar2d_result_filtering(cfar_indices)
        # print(f"CFAR结果:\r\n{cfar_indices}")

        # 计算角度、速度和距离
        measurements = self.__calc_measurement(cfar_indices, rdms)
        # print(f"测量值：\r\n{meas}")

        # 聚类
        self.basic.multi_frame_meas.append(measurements)
        measurements = self.__cluster(eps=0.7, min_samples=5)
        self.basic.measurements = measurements

        # 跟踪
        _measurements = [np.vstack(([0], x[:3].reshape(3, 1))) for x in measurements]
        track(associator, _deleter, _initiator, tracked_targets, unconfirmed_targets, _measurements, 5, timestamp)

        return measurements

    def __cfar2d_goca(self):
        cfg = self.config.cfar_cfg
        mag = self.basic.mag
        _, noise = cfar_2d(self.basic.mag, cfg.numTrain, cfg.numGuard, threshold=cfg.thSNR, type="GOCA")
        snr = mag / noise
        indices = np.argwhere(np.logical_and(snr > cfg.thSNR, mag > cfg.thMag))
        ans = [CFAR2dPoint(i[0], i[1], mag[i[0], i[1]], snr[i[0], i[1]]) for i in indices if i[0] != 0]  # 去除零距离
        return ans

    def __cfar2d_result_filtering(self, cfar_indices):
        cfg = self.config.cfar_filter_cfg
        return cfar_result_filtering(cfar_indices, range0=cfg.range0, range1=cfg.range1, shape1=cfg.shape1, th=cfg.th)

    def __calc_measurement(self, cfar_indices, rdm):
        if len(cfar_indices) == 0:
            return []
        mag = self.basic.mag
        x = np.array(cfar_indices)
        idx0 = x[:, 0].astype(int)
        idx1 = x[:, 1].astype(int)
        delta_phase = (np.angle(rdm[0, idx0, idx1]) - np.angle(rdm[1, idx0, idx1]) + np.pi) % (2 * np.pi) - np.pi
        phi = np.arcsin(delta_phase / np.pi)

        ret = []
        for item, p in zip(cfar_indices, phi):
            item: CFAR2dPoint
            if np.abs(p) < self.config.channel_phase_diff_threshold:
                idxR = round(item.idx0)
                idxV = round(item.idx1)
                if idxV > mag.shape[1] // 2:
                    velo = (idxV - mag.shape[1]) * self.param.resVelocity
                else:
                    velo = idxV * self.param.resVelocity

                a = idxR
                if a == 0:
                    a = 1
                    b = -1
                elif a + 1 == mag.shape[0] or mag[a - 1, idxV] > mag[a + 1, idxV]:
                    b = -1
                elif mag[a - 1, idxV] < mag[a + 1, idxV]:
                    b = 1
                else:
                    b = 0
                d = mag[a + b, idxV]

                if d != 0:
                    r = mag[a, idxV] / d
                    idxR = a + b / r
                dis = idxR * self.param.resRange
                ret.append(Measurement(p, dis, velo, item.mag, item.snr))
        return ret

    def __cluster(self, eps=0.7, min_samples=5):
        X = np.array([m for measurements in self.basic.multi_frame_meas for m in measurements])
        _X = np.empty(shape=(len(X), 3))
        _X[:, 0] = X[:, 1] * np.cos(X[:, 0])
        _X[:, 1] = X[:, 1] * np.sin(X[:, 0])
        _X[:, 2] = X[:, 2]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(_X)
        num_cluster = np.max(labels) + 1
        return [Measurement(np.mean(X[labels == i], axis=0)) for i in range(num_cluster)]


# %% 加载数据

file = scipy.io.loadmat(file_name="../data/AT24G_RecordedData_丰海条形模块_单人行走.mat")
radar_data_cube = file["RDM"][1:].transpose(0, 1, 3, 2)
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

numTrain = (3, 8)
numGuard = (2, 4)
del file


# %%


timestamp_start = datetime.datetime.now()

# 初始化
processor = Processor(
    param=Radarparam(
        numChrip=numChrip,
        numRangeBin=numRangeBin,
        numChannel=numChannel,
        lambda_over_d=1.78,
        resRange=0.71,
        resVelocity=resVelocity,
    ),
    config=RadarConfig(
        cfar_cfg=RadarCFARConfig(numTrain=numTrain, numGuard=numGuard, thSNR=3.5, thMag=500),
        cfar_filter_cfg=RadarCFARFilterConfig(range0=2, range1=3, shape1=64, th=0.8),
        channel_phase_diff_threshold=np.pi * 0.98,
    ),
)

meas = processor(radar_data_cube[0], timestamp_start)


# %%
from stonesoup.types.state import GaussianState
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.types.detection import Detection

measurement_list = []

all_measurements = []

trajectorys = dict()
unconfirmed_trajectorys = dict()

timestamps = []
for i, rdm in enumerate(radar_data_cube):
    timestamp = timestamp_start + datetime.timedelta(seconds=i * timeFrame)
    timestamps.append(timestamp)
    m = processor(rdm, timestamp)

    # 保存测量值
    measurement_list.append(m)
    measurement_set = set()
    for m in processor.basic.measurements:
        x = m[1] * np.cos(m[0])
        y = m[1] * np.sin(m[0])
        measurement_set.add(Detection(state_vector=[x, y], timestamp=timestamp))
    all_measurements.append(measurement_set)

    # 保存轨迹，用于绘图
    for target in tracked_targets:
        target: TrackedTarget
        uuid = target.uuid
        if uuid in trajectorys.keys():
            trajectorys[uuid].append(GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp))
        else:
            trajectorys[uuid] = [GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp)]
    for target in unconfirmed_targets:
        target: TrackedTarget
        uuid = target.uuid
        if uuid in unconfirmed_trajectorys.keys():
            unconfirmed_trajectorys[uuid].append(GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp))
        else:
            unconfirmed_trajectorys[uuid] = [GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp)]

# %%
# 绘制点云

listData = genScatterPolar(measurement_list, tail_length=5, name="点云")
fig = dh.draw_animation(listData, title="超分辨率后点云")
fig.update_layout(
    xaxis=dict(title="前后", range=[0, 10], scaleanchor="y", scaleratio=1, constrain="domain"),
    yaxis=dict(title="左右", range=[-5, 5], scaleanchor="x", scaleratio=1, constrain="domain"),
)
fig.show()

# %%
plotter = AnimatedPlotterly(timestamps, tail_length=0.12)
plotter.plot_measurements(all_measurements, [0, 1], convert_measurements=False)
plotter.plot_tracks(trajectorys.values(), [0, 2], track_label="User-confirmed")
plotter.plot_tracks(unconfirmed_trajectorys.values(), [0, 2], track_label="User-unconfirmed", marker=dict(symbol="x", size=8))
plotter.fig
# %%
go.Figure(data=go.Surface(z=np.sum(np.abs(radar_data_cube[104]), axis=0)))

# %%
import sys

sys.path.append("../")


import numpy as np
import scipy.constants
import scipy.io
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

from sklearn.cluster import DBSCAN
from collections import deque

from myRadar.cfar import cfar_2d, cfar_result_filtering
from myRadar.base import BaseBasicData


from itertools import chain


import datetime


# %%
# 跟踪器

# 自己实现的GNN目标跟踪
from scipy.optimize import linear_sum_assignment

import myRadar.track.kalman as mk
import myRadar.track.models as mm
from myRadar.base import Printable


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

    def __init__(self, measurement, init_covar, timestamp: datetime):
        TrackedTarget.next_uuid += 1

        self.uuid = TrackedTarget.next_uuid
        self.state = self.__clac_init_state(measurement, init_covar, timestamp)
        self.life_cycle = LifeCycle(Initiator.initial_score)
        self.life_cycle.measurements.append(measurement)
        self.life_cycle.post_measurements.append(measurement)

    def __clac_init_state(self, measurement, init_covar, timestamp):
        theta, phi, rho, rho_rate = measurement
        x = rho * np.cos(theta) * np.cos(phi)
        y = rho * np.cos(theta) * np.sin(phi)
        z = rho * np.sin(theta)
        vx = rho_rate * np.cos(theta) * np.cos(phi)
        vy = rho_rate * np.cos(theta) * np.sin(phi)
        vz = rho_rate * np.sin(theta)
        state = mk.GaussianState(state_vector=np.array([x, vx, y, vy, z, vz]).reshape(-1, 1), covar=init_covar, timestamp=timestamp)
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

        # 最小和指派
        row4col, col4row = linear_sum_assignment(distance_matrix, 0)

        # 给每一个假设分配测量值
        for i, j in zip(row4col, col4row):
            if distance_matrix[i, j] != np.inf and j < len(measurements):
                hypotheses[row_map[i]].measurement = measurements[j]
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
        init_covar: np.ndarray = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) ** 2,
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
        self.init_covar = init_covar

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
            unconfirmed_targets.append(TrackedTarget(measurement, init_covar=self.init_covar, timestamp=timestamp))


class Deleter(Printable):
    max_score = 5000

    def __init__(
        self,
        updater: mk.KalmanUpdater,
        unassociated_time=10.0,  # 目标关联失败超时时间
        missed_probability=0.2,  # 目标丢失的概率
        fov=np.array([-np.pi / 3, np.pi / 3]),  # 视角
        radius_range=np.array([0.2, 100]),  # 半径范围
    ):
        self.updater = updater

        self.unassociated_time = unassociated_time
        self.unassociated_score = int(-Deleter.max_score / unassociated_time)
        self.missed_probability = missed_probability
        self.radius_range = radius_range
        self.fov = fov

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
            score -= self.unassociated_score * dt * self.missed_probability / (1 - self.missed_probability)
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

        # 离开监测范围的目标需要删除
        angle = np.arctan2(post.state_vector[2], post.state_vector[0])
        r = np.linalg.norm([post.state_vector[0], post.state_vector[2]])
        if (angle > self.fov[1]) or (angle < self.fov[0]):
            life_cycle.score = -1
        if r > self.radius_range[1] or r < self.radius_range[0]:
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


class Tracker:

    def __init__(
        self,
        tran_model_q,
        meas_noise_r,
        del_unassociated_time,
        del_missed_probability,
        init_unassociated_time,
        init_keep_motion_time,
        init_keep_static_time,
        init_speed_th,
        init_missed_distance,
        init_covar,
        fov,
        radius_range,
    ):
        self.preictor = mk.KalmanPredictor(mm.TransitionModel(tran_model_q))
        self.updater = mk.KalmanUpdater(mm.MeasurementModel(meas_noise_r))
        self.associator = Associator(self.preictor, self.updater)
        self.deleter = Deleter(
            updater=self.updater, unassociated_time=del_unassociated_time, missed_probability=del_missed_probability, fov=fov, radius_range=radius_range
        )
        self.initiator = Initiator(
            predictor=self.preictor,
            updater=self.updater,
            associator=self.associator,
            unassociated_time=init_unassociated_time,
            keep_motion_time=init_keep_motion_time,
            keep_static_time=init_keep_static_time,
            speed_threshold=init_speed_th,
            missed_distance=init_missed_distance,
            init_covar=init_covar,
        )

    def track(self, tracked_targets, unconfirmed_targets, measurements, missed_distance, timestamp):
        # 数据关联，得到假设和未关联的测量
        hypotheses, unassociated_measurements = self.associator.associate(tracked_targets, measurements, timestamp, missed_distance)

        # 删除无效目标
        self.deleter.delete(tracked_targets, hypotheses)
        # for target in tracked_targets:
        #     print(f"跟踪目标 {target.uuid} 误差 {target.life_cycle.mese} 分数 {target.life_cycle.score}")

        # 航迹起始
        self.initiator.initiate(tracked_targets, unconfirmed_targets, unassociated_measurements, timestamp)


# %%
# 处理器


class CFAR2dPoint(BaseBasicData):
    attributes = ["idx0", "idx1", "mag", "snr"]


class Measurement(BaseBasicData):
    attributes = ["azimuth", "distance", "velocity", "mag", "snr"]


from dataclasses import dataclass


@dataclass
class RadarInitParam:
    wavelength: float  # 波长 (m)
    bandwidth: float  # 带宽 (Hz)
    rx_antenna_spacing: float  # 接收天线间距 (m)
    timeChrip: float  # Chrip 调频时长 (s)
    timeChripGap: float  # Chrip 间距，从一个 Chrip 结束到下一个 Chrip 开始 (s)
    timeFrameGap: float  # 帧间距，从一个帧结束到下一个帧开始 (s)
    numChannel: int  # 雷达通道数
    numRangeBin: int  # 距离单元数
    numChrip: int  # Chrip 数
    numMaxCfarPoints: int  # CFAR 检测的最大点数，超出上限时较远距离的点会被丢弃
    numMaxCachedFrame: int  # 缓存帧的最大数量，用于叠加多帧聚类
    numInitialMultiMeas: int  # 缓存多帧量测值的数组的初始大小
    numInitialCluster: int  # 缓存聚类结果的数组的初始大小


@dataclass
class RadarParam:
    # 输入参数
    wavelength: float  # 单位: m, 雷达波长，例如 24GHz 雷达波长为 12.42663038e-3
    bandwidth: float  # 单位: Hz, 雷达有效带宽
    timeChrip: float  # 单位: s, 每个 Chrip 的时间
    timeChripGap: float  # 单位: s, Chrip 间隔
    timeFrameGap: float  # 单位: s, 帧间隔

    numChannel: int  # 雷达通道数
    numSample: int  # 采样点数
    numRangeBin: int  # 距离单元数量
    numChrip: int  # Chrip 数

    # 衍生参数
    timeChripTotal: float  # 单位: s, 一个完整 Chrip 的时间 (timeChrip + timeChripGap)
    timeFrame: float  # 单位: s, 一帧的有效时间 (numChrip * timeChripFull)
    timeFrameTotal: float  # 单位: s, 一帧的总时间 (timeFrameValid + timeFrameGap)
    resRange: float  # 单位: m, 距离分辨率
    resVelocity: float  # 单位: m/s, 速度分辨率

    lambda_over_d: float  # 波长/天线间


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
class DBSCANConfig:
    wr: float
    wv: float
    eps: float
    min_samples: int


@dataclass
class RadarCFARFilterConfig:
    range0: int
    range1: int
    shape1: int
    th: float


@dataclass
class TrackConfig:
    tran_model_q: float
    meas_noise_r: np.ndarray
    missed_distance: float
    del_unassociated_time: float
    del_missed_probability: float
    init_unassociated_time: float
    init_keep_motion_time: float
    init_keep_static_time: float
    init_speed_th: float
    init_missed_distance: float
    init_covar: np.ndarray
    fov: np.ndarray
    radius_range: np.ndarray


@dataclass
class RadarConfig:
    cfar_cfg: RadarCFARConfig
    cfar_filter_cfg: RadarCFARFilterConfig
    dbscan_cfg: DBSCANConfig
    track_cfg: TrackConfig
    channel_phase_diff_threshold: float


# %%


class Processor:

    def __init__(self, param: RadarInitParam, config: RadarConfig):
        timeChripTotal = param.timeChrip + param.timeChripGap
        timeFrame = timeChripTotal * param.numChrip
        timeFrameTotal = timeFrame + param.timeFrameGap
        self.param = RadarParam(
            wavelength=param.wavelength,
            bandwidth=param.bandwidth,
            timeChrip=param.timeChrip,
            timeChripGap=param.timeChripGap,
            timeFrameGap=param.timeFrameGap,
            numChannel=param.numChannel,
            numSample=param.numRangeBin * param.numChrip,
            numRangeBin=param.numRangeBin,
            numChrip=param.numChrip,
            timeChripTotal=timeChripTotal,
            timeFrame=timeFrame,
            timeFrameTotal=timeFrameTotal,
            resRange=scipy.constants.c / (2 * param.bandwidth),
            resVelocity=param.wavelength / (2 * timeFrame),
            lambda_over_d=param.wavelength / param.rx_antenna_spacing,
        )
        self.basic = RadarBasicData(
            mag=np.zeros(shape=(param.numRangeBin, param.numChrip)),
            multi_frame_meas=deque(maxlen=param.numMaxCachedFrame),
            measurements=[],
        )

        self.config = config

        self.tracker = Tracker(
            tran_model_q=config.track_cfg.tran_model_q,
            meas_noise_r=config.track_cfg.meas_noise_r,
            del_unassociated_time=config.track_cfg.del_unassociated_time,
            del_missed_probability=config.track_cfg.del_missed_probability,
            init_unassociated_time=config.track_cfg.init_unassociated_time,
            init_keep_motion_time=config.track_cfg.init_keep_motion_time,
            init_keep_static_time=config.track_cfg.init_keep_static_time,
            init_speed_th=config.track_cfg.init_speed_th,
            init_missed_distance=config.track_cfg.init_missed_distance,
            init_covar=config.track_cfg.init_covar,
            fov=config.track_cfg.fov,
            radius_range=config.track_cfg.radius_range,
        )

        self.tracked_targets = []  # 被跟踪的目标
        self.unconfirmed_targets = []  # 航迹起始阶段的目标

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

        # 聚类
        self.basic.multi_frame_meas.append(measurements)
        measurements = self.__cluster()
        self.basic.measurements = measurements

        # 跟踪
        _measurements = [np.vstack(([0], x[:3].reshape(3, 1))) for x in measurements]
        self.tracker.track(
            tracked_targets=self.tracked_targets,
            unconfirmed_targets=self.unconfirmed_targets,
            measurements=_measurements,
            missed_distance=self.config.track_cfg.missed_distance,
            timestamp=timestamp,
        )

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
        phi = np.arcsin(delta_phase * self.param.lambda_over_d / (2 * np.pi))

        ret = []
        for item, phase, p in zip(cfar_indices, delta_phase, phi):
            item: CFAR2dPoint
            if np.abs(phase) < self.config.channel_phase_diff_threshold:
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

    def __cluster(self):
        cfg = self.config.dbscan_cfg
        wr = cfg.wr
        wv = cfg.wv
        eps = cfg.eps
        min_samples = cfg.min_samples
        X = np.array(list(chain.from_iterable(self.basic.multi_frame_meas)))
        if X.ndim == 1:
            return []
        _X = np.empty(shape=(len(X), 3))
        _X[:, 0] = X[:, 1] * np.cos(X[:, 0]) * wr
        _X[:, 1] = X[:, 1] * np.sin(X[:, 0]) * wr
        _X[:, 2] = X[:, 2] * wv
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(_X)
        num_cluster = np.max(labels) + 1
        return [Measurement(np.mean(X[labels == i], axis=0)) for i in range(num_cluster)]

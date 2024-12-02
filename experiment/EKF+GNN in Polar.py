"""
对于单个目标来说，目标跟踪分三个阶段： 航迹起始，航迹关联，航迹结束

航迹起始阶段需要排除一些杂波的干扰，在起始延迟和虚假率之间进行平衡

航迹关联算法需要解决多目标关联问题，以及关联失败时的预测。常用的多目标数据关联算法有全局最近邻(GNN)、联合概率数据关联(JPDA)、多假设跟踪(MHT)等。

航迹结束阶段需要判断目标是否已经消失，目标走出ROI，以及关联失败的时间太长或者误差协方差太大时删除tracker

对于整个系统来说，每一帧都要执行上文的三个步骤，完成多目标跟踪，其中数据关联部分步骤大致如下：

1. 预测：使用k-1时刻的状态向量预测k时刻的状态向量，格局测量模型将预测状态向量转化为预测测量值 
    - x_k|k-1 = f(x_k-1|k-1)
    - z_k|k-1 = h(x_k|k-1)
2. 关联：匹配预测测量值和实际测量值

3. 更新：将新匹配的测量值添加到tracker中，通过滤波算法得到纠正后的状态向量 x_k|k



"""

# %%
import sys

sys.path.append("../")

import numpy as np
import scipy.io
from datetime import datetime, timedelta
from stonesoup.plotter import AnimatedPlotterly
import scipy.interpolate

# %%
# 导入轨迹
posScale = 10
stride = 10
num_steps = 200
axisTime = np.linspace(0, 60, num_steps, endpoint=False)

file_list = [
    "../data/mouse_trajectory_2024_09_13_17_22_54.mat",
    "../data/mouse_trajectory_2024_09_13_17_23_04.mat",
    "../data/mouse_trajectory_2024_12_01_23_48_29.mat",
]

targets_info = []
for file in file_list:
    trajTemp = scipy.io.loadmat(file)
    targets_info.append([trajTemp["timestamps"].reshape(-1), trajTemp["positions"]])


targets_position = []
for times, pos in targets_info:
    trajTemp = scipy.io.loadmat(file)
    interp_pos = scipy.interpolate.interp1d(
        times * (np.max(axisTime) / np.max(times)),
        pos,
        axis=0,
        kind="quadratic",
        bounds_error=False,
        fill_value=(trajTemp["positions"][0], trajTemp["positions"][-1]),
    )
    targets_position.append(interp_pos(axisTime) * posScale)


# 设置参数
clutter_num_max = 1
velocity_noise_coef = 1.5
meas_noise_covar = np.diag([0, 5 / 180 * np.pi, 0.2, 0.01]) ** 2
missed_distance = 10

# %%
# 设置状态转移模型
# 状态向量[x,vx,y,vy,z,vz]，匀速运动模型，三个维度互相独立

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(velocity_noise_coef), ConstantVelocity(velocity_noise_coef), ConstantVelocity(velocity_noise_coef)]
)

# 设置测量模型
# 测量向量为 [phi,r,vr] 即[方位角，径向距离，径向速度]
# z = Hx ， x为状态向量，H为测量矩阵，z为测量值

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate


measurement_model = CartesianToElevationBearingRangeRate(
    ndim_state=6,
    mapping=[0, 2, 4],
    noise_covar=meas_noise_covar,
)


# %%
# 生成时间戳
start_time = datetime.now().replace(microsecond=0)

timestamps = np.array([start_time + timedelta(seconds=t) for t in axisTime])

# 生成真值轨迹
# 真值是状态向量的格式[x,y,vx,vy]，笛卡尔坐标系
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from ordered_set import OrderedSet

truths = OrderedSet()
for i, pos in enumerate(targets_position):
    stateVectors = np.zeros((num_steps, 6))
    stateVectors[:num_steps, :3] = pos[:num_steps, :3]
    dt = np.array([t.total_seconds() for t in (timestamps[1:] - timestamps[:-1])]).reshape(-1, 1)
    stateVectors[:-1, 3:6] = (stateVectors[1:, 0:3] - stateVectors[:-1, 0:3]) / dt
    stateVectors = stateVectors[:-1, [0, 3, 1, 4, 2, 5]]
    states = [GroundTruthState(x, t) for x, t in zip(stateVectors, timestamps)]
    truths.add(GroundTruthPath(states=states, id=i))

# 生成带噪声的测量值，作为仿真的输入
from scipy.stats import uniform, norm

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

all_measurements = []

for k in range(len(stateVectors)):
    measurement_set = set()

    for truth in truths:
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= 1:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(
                TrueDetection(state_vector=measurement, groundtruth_path=truth, timestamp=truth[k].timestamp, measurement_model=measurement_model)
            )

        # 生成速度为0的杂点，在真实目标背后
        truth_theta, truth_phi, truth_rho, truth_rho_rate = measurement_model.function(truth[k])
        for _ in range(np.random.randint(clutter_num_max + 1)):
            theta = truth_theta
            phi = uniform.rvs(truth_phi - 0.2, 0.4)
            rho = uniform.rvs(truth_rho + 1, 5)
            rho_rate = 0
            measurement_set.add(Clutter(np.array([[theta], [phi], [rho], [rho_rate]]), timestamp=truth[k].timestamp, measurement_model=measurement_model))
    all_measurements.append(measurement_set)


# %%
# 绘图


# plotter = AnimatedPlotterly(timestamps, tail_length=0.2)
# plotter.plot_ground_truths(truths, [0, 2])
# plotter.plot_measurements(all_measurements, [0, 2])
# plotter.fig


# %%
# 创建拓展卡尔曼滤波器
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# 创建GNN数据关联器
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour, GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Measure, EuclideanWeighted
from scipy.spatial import distance


class EuclideanBearingRangeRate(Measure):
    def __call__(self, state1, state2):
        # Calculate Euclidean distance between two state
        state_vector1 = getattr(state1, "mean", state1.state_vector)
        state_vector2 = getattr(state2, "mean", state2.state_vector)
        if self.mapping is not None:
            state_vector1 = state_vector1[self.mapping]
            state_vector2 = state_vector2[self.mapping2]
        state_vector1 = np.array([state_vector1[1] * np.cos(state_vector1[0]), state_vector1[1] * np.sin(state_vector1[0]), state_vector1[2]])
        state_vector2 = np.array([state_vector2[1] * np.cos(state_vector2[0]), state_vector2[1] * np.sin(state_vector2[0]), state_vector2[2]])
        ret = distance.euclidean(state_vector1, state_vector2)
        return ret


# 极坐标转化成直角坐标后计算欧式距离，即圆形波门
# hypothesiser = DistanceHypothesiser(predictor, updater, measure=EuclideanBearingRangeRate(mapping=[1, 2, 3]), missed_distance=10)

# 直接将[角度，径向距离，径向速度]加权后计算欧式距离，即扇形波门
hypothesiser = DistanceHypothesiser(predictor, updater, measure=EuclideanWeighted(weighting=(0, 2, 1, 1.5)), missed_distance=5)

data_associator = GNNWith2DAssignment(hypothesiser)

# 创建删除器
from stonesoup.deleter.error import CovarianceBasedDeleter

deleter = CovarianceBasedDeleter(covar_trace_thresh=4)

# 创建启动器
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator

initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0], [0], [0]], np.diag([1, 0.5, 1, 0.5, 1, 0.5])),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=2,
)


# %%
# 初始化tracker
from stonesoup.types.state import GaussianState

tracks, all_tracks = set(), set()

for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks, measurements, timestamps[n])
    associated_measurements = set()
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
            associated_measurements.add(hypothesis.measurement)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

    # Carry out deletion and initiation
    tracks -= deleter.delete_tracks(tracks)
    possiable_measurements = {i for i in (measurements - associated_measurements) if abs(i.state_vector[3]) > 0.02}
    tracks |= initiator.initiate(possiable_measurements, timestamps[n])
    all_tracks |= tracks
# %%
# 绘制跟踪结果


# plotter = AnimatedPlotterly(timestamps, tail_length=0.2)
# plotter.plot_ground_truths(truths, [0, 2])
# plotter.plot_measurements(all_measurements, [0, 2])
# plotter.plot_tracks(tracks, [0, 2])
# plotter.fig


# %%
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


class myTracker:
    next_uuid = 0

    def __init__(self, measurement):
        self.uuid = myTracker.next_uuid
        myTracker.next_uuid += 1

        theta, phi, rho, rho_rate = measurement
        x = rho * np.cos(theta) * np.cos(phi)
        y = rho * np.cos(theta) * np.sin(phi)
        z = rho * np.sin(theta)
        vx = rho_rate * np.cos(theta) * np.cos(phi)
        vy = rho_rate * np.cos(theta) * np.sin(phi)
        vz = rho_rate * np.sin(theta)
        state = mk.GaussianState(state_vector=np.array([x, vx, y, vy, z, vz]).reshape(-1, 1), covar=np.diag([1, 0.5, 1, 0.5, 1, 0.5]) ** 2, timestamp=timestamp)

        self.state = state
        self.life_cycle = LifeCycle(Initiator.initial_score)
        self.life_cycle.measurements.append(measurement)
        self.life_cycle.post_measurements.append(measurement)


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
        dt = hypothesis.prediction.timestamp - hypothesis.prior_state.timestamp
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
            target: myTracker
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
            unconfirmed_targets.append(myTracker(measurement))


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
        dt = hypothesis.prediction.timestamp - hypothesis.prior_state.timestamp
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

    # 按照假设更新每一个目标
    for target, hypothesis in zip(tracked_targets, hypotheses):
        if hypothesis.measurement is None:
            target.state = hypothesis.prediction
        else:
            target.state = ekf_updater.update(hypothesis)

    # 删除无效目标
    deleter.delete(tracked_targets, hypotheses)

    # 航迹起始
    initiator.initiate(tracked_targets, unconfirmed_targets, unassociated_measurements, timestamp)

    for i in tracked_targets:
        print(f"跟踪目标 {i.uuid} 误差 {i.life_cycle.mese} 分数 {i.life_cycle.score}")
    for i in unconfirmed_targets:
        print(f"航迹起始 {i.uuid} 误差 {i.life_cycle.mese} 分数 {i.life_cycle.score}")


# %%

# 创建工作类  预测、更新、关联、起始
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

#  轨迹
trajectorys = dict()
unconfirmed_trajectorys = dict()

# 目标
tracked_targets = []  # 被跟踪的目标
unconfirmed_targets = []  # 航迹起始阶段的目标

for n, measurements in enumerate(all_measurements[:]):
    print(f"\r\n\r\n时间 {timestamps[n]}")
    measurements = [m.state_vector.astype(np.float64) for m in measurements]
    timestamp = (timestamps[n] - timestamps[0]).total_seconds()

    # 跟踪
    track(associator, _deleter, _initiator, tracked_targets, unconfirmed_targets, measurements, missed_distance, timestamp)

    # 保存轨迹，用于绘图
    for target in tracked_targets:
        target: myTracker
        uuid = target.uuid
        if uuid in trajectorys.keys():
            trajectorys[uuid].append(GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamps[n]))
        else:
            trajectorys[uuid] = [GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamps[n])]
    for target in unconfirmed_targets:
        target: myTracker
        uuid = target.uuid
        if uuid in unconfirmed_trajectorys.keys():
            unconfirmed_trajectorys[uuid].append(GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamps[n]))
        else:
            unconfirmed_trajectorys[uuid] = [GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamps[n])]


# %%
plotter = AnimatedPlotterly(timestamps, tail_length=0.12)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(tracks, [0, 2], track_label="Stonesoup")
plotter.plot_tracks(trajectorys.values(), [0, 2], track_label="User-confirmed")
plotter.plot_tracks(unconfirmed_trajectorys.values(), [0, 2], track_label="User-unconfirmed", marker=dict(symbol="x", size=8))
plotter.fig
# %%


# %%

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

# %%
# 导入轨迹
timeScale = 15
posScale = 10
stride = 20
file_list = ["../data/mouse_trajectory_2024_09_13_17_22_54.mat", "../data/mouse_trajectory_2024_09_13_17_23_04.mat"]

targetsInfo = []
for file in file_list:
    trajTemp = scipy.io.loadmat(file)
    targetsInfo.append(dict(rsc=1, times=trajTemp["timestamps"].ravel()[::stride] * timeScale, pos=trajTemp["positions"][::stride] * posScale))
num_steps = min([len(i["times"]) for i in targetsInfo])

# 设置参数
clutter_num_max = 2


# %%
# 设置状态转移模型
# 状态向量[x,vx,y,vy,z,vz]，匀速运动模型，三个维度互相独立

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5), ConstantVelocity(0.5), ConstantVelocity(0.5)])

# 设置测量模型
# 测量向量为 [phi,r,vr] 即[方位角，径向距离，径向速度]
# z = Hx ， x为状态向量，H为测量矩阵，z为测量值

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

measurement_model = CartesianToElevationBearingRangeRate(
    ndim_state=6,
    mapping=[0, 2, 4],
    noise_covar=np.diag([0, 5 / 180 * np.pi, 0.2, 0.01]) ** 2,
)

from scipy.linalg import inv
from stonesoup.functions import sphere2cart
from types import MethodType


# %%
# 生成时间戳
start_time = datetime.now().replace(microsecond=0)

timesteps = [start_time + timedelta(seconds=t) for t in targetsInfo[0]["times"][:num_steps]]

# 生成真值轨迹
# 真值是状态向量的格式[x,y,vx,vy]，笛卡尔坐标系
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from ordered_set import OrderedSet

truths = OrderedSet()
for i, target in enumerate(targetsInfo):
    timestamps = target["times"][:num_steps]
    stateVectors = np.zeros((num_steps, 6))
    stateVectors[:num_steps, :3] = target["pos"][:num_steps, :3]
    stateVectors[:-1, 3:6] = (stateVectors[1:, 0:3] - stateVectors[:-1, 0:3]) / (timestamps[1:] - timestamps[:-1]).reshape(-1, 1)
    stateVectors = stateVectors[:-1, [0, 3, 1, 4, 2, 5]]
    states = [GroundTruthState(x, start_time + timedelta(seconds=t)) for x, t in zip(stateVectors, timestamps)]
    truths.add(GroundTruthPath(states=states, id=i))

# 生成带噪声的测量值，作为仿真的输入
from scipy.stats import uniform, norm

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian

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
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.2)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.fig


# %%
# 创建拓展卡尔曼滤波器
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# 创建GNN数据关联器
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
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

data_associator = GlobalNearestNeighbour(hypothesiser)

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
    min_points=5,
)


# %%
# 初始化tracker
from stonesoup.types.state import GaussianState

tracks, all_tracks = set(), set()

for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks, measurements, timesteps[n])
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
    tracks |= initiator.initiate(possiable_measurements, timesteps[n])
    all_tracks |= tracks
# %%
# 绘制跟踪结果
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.2)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(tracks, [0, 2])
plotter.fig

# %%

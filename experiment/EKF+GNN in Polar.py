"""
跟踪步骤

1. 预测：使用k-1时刻的状态向量预测k时刻的状态向量 x_k|k-1 = f(x_k-1|k-1)
2. 关联：

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
noise_num_max = 3
noise_radius = 0.2

# %%
# 生成时间戳
start_time = datetime.now().replace(microsecond=0)

timesteps = [start_time + timedelta(seconds=t) for t in targetsInfo[0]["times"][:num_steps]]
# %%
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

# %%
# 绘制真值轨迹
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig


# %% 设置运动模型

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.3), ConstantVelocity(0.3), ConstantVelocity(0)])


# %%
# 设置测量模型
# 测量向量为 [phi,r,vr] 即[方位角，径向距离，径向速度]
# z = Hx ， x为状态向量，H为测量矩阵，z为测量值
from stonesoup.models.measurement.nonlinear import CartesianToBearingRangeRate
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

measurement_model = CartesianToBearingRangeRate(
    ndim_state=6,
    mapping=[0, 2, 4],
    noise_covar=np.diag([5 / 180 * np.pi, 0.2, 0.01]) ** 2,
)

from scipy.linalg import inv
from stonesoup.functions import sphere2cart
from types import MethodType


def inverse_function(self, detection, **kwargs) -> StateVector:
    phi, rho, rho_rate = detection.state_vector
    theta = 0

    x, y, z = sphere2cart(rho, phi, theta)

    x_rate, y_rate, z_rate = sphere2cart(rho_rate, phi, theta)

    inv_rotation_matrix = inv(self.rotation_matrix)

    out_vector = StateVector([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    out_vector[self.mapping, 0] = x, y, z
    out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

    out_vector[self.mapping, :] = inv_rotation_matrix @ out_vector[self.mapping, :]
    out_vector[self.velocity_mapping, :] = inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

    out_vector[self.mapping, :] = out_vector[self.mapping, :] + self.translation_offset
    out_vector[self.velocity_mapping, :] = out_vector[self.velocity_mapping, :] + self.velocity

    return out_vector


# 添加inverse_function方法到measurement_model，用于将测量值转化为状态向量
measurement_model.inverse_function = MethodType(inverse_function, measurement_model)


# %%
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
        truth_phi, truth_rho, truth_rho_rate = measurement_model.function(truth[k])
        for _ in range(np.random.randint(noise_num_max)):
            phi = uniform.rvs(truth_phi - 0.2, 0.4)
            rho = uniform.rvs(truth_rho + 1, 5)
            rho_rate = 0
            measurement_set.add(Clutter(np.array([[phi], [rho], [rho_rate]]), timestamp=truth[k].timestamp, measurement_model=measurement_model))
    all_measurements.append(measurement_set)

# %%
# 绘制测量值
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
# hypothesiser = DistanceHypothesiser(predictor, updater, measure=EuclideanBearingRangeRate(), missed_distance=10)

# 直接将[角度，径向距离，径向速度]加权后计算欧式距离，即扇形波门
hypothesiser = DistanceHypothesiser(predictor, updater, measure=EuclideanWeighted(weighting=(0.5, 1, 1)), missed_distance=10)

data_associator = GlobalNearestNeighbour(hypothesiser)


# %%
# 初始化tracker
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

tracks = [Track([GaussianState(truth[0].state_vector, np.diag([1, 0.1, 2, 0.2, 0, 0]) ** 2, timestamp=start_time)]) for truth in truths]

# 全局最近邻跟踪
for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks, measurements, timesteps[n])
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            print(f"{timesteps[n]} {hypothesis.distance}")
            post = updater.update(hypothesis)
            track.append(post)
        else:  # 匹配失败时，使用预测值作为新的状态向量
            prediction = hypothesis.prediction
            prediction.state_vector[[1, 3, 5]] /= 2
            track.append(hypothesis.prediction)
            print(f"{ timesteps[n]} no detection, {hypothesis.prediction.state_vector}")
# %%
# 绘制跟踪结果
plotter = AnimatedPlotterly(timesteps, tail_length=0.2)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(tracks, [0, 2])
plotter.fig

# %%

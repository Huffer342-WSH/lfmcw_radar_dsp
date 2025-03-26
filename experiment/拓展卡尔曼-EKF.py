"""


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
from math import sqrt

np.set_printoptions(precision=3, suppress=True)

# %%
# 导入轨迹
timeScale = 0.1
posScale = 10
stride = 5
file = "../data/mouse_trajectory_2024_09_13_17_22_54.mat"


target_pos = scipy.io.loadmat(file)["positions"][::stride] * posScale
# target_pos = target_pos[:20]
num_steps = len(target_pos)
start_time = datetime(2000, 1, 1, 0, 0, 0, 0)
timesteps = np.array([start_time + timedelta(seconds=i * timeScale) for i in range(1, num_steps + 1)])

truth_state_vector = np.zeros((num_steps, 6))
truth_state_vector[:, :3] = target_pos[:, :3]
truth_state_vector[:-1, 3:6] = (truth_state_vector[1:, 0:3] - truth_state_vector[:-1, 0:3]) / timeScale
truth_state_vector = truth_state_vector[:-1, [0, 3, 1, 4, 2, 5]]


# 设置参数
q = 1
P0 = np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5])
x0 = truth_state_vector[0].reshape(-1, 1)
R0 = np.diag([0, 5 / 180 * np.pi, 0.2, 0.01]) ** 2
z_lizt = []

# %%
# 设置状态转移模型
# 状态向量[x,vx,y,vy,z,vz]，匀速运动模型，三个维度互相独立

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q), ConstantVelocity(q), ConstantVelocity(q)])

# 设置测量模型
# 测量向量为 [phi,r,vr] 即[方位角，径向距离，径向速度]
# z = Hx ， x为状态向量，H为测量矩阵，z为测量值

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

measurement_model = CartesianToElevationBearingRangeRate(
    ndim_state=6,
    mapping=[0, 2, 4],
    noise_covar=R0,
)

from scipy.linalg import inv
from stonesoup.functions import sphere2cart
from types import MethodType


# 生成真值轨迹
# 真值是状态向量的格式[x,y,vx,vy]，笛卡尔坐标系
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

states = [GroundTruthState(x, t) for x, t in zip(truth_state_vector, timesteps)]
truth = GroundTruthPath(states=states)

# 生成带噪声的测量值，作为仿真的输入
from scipy.stats import uniform, norm

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Detection


measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp, measurement_model=measurement_model))


# %%
# 绘图
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.2)
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2])
plotter.fig


# %%
# 创建拓展卡尔曼滤波器
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


prior = GaussianState(x0, P0, timestamp=start_time)
track = Track()
post = prior
for measurement in measurements:
    prediction = predictor.predict(post, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    post = track[-1]


# %%
import myRadar.track.kalman as mk
import myRadar.track.models as mm

efk_predictor = mk.KalmanPredictor(mm.TransitionModel(q))
efk_updater = mk.KalmanUpdater(mm.MeasurementModel(R0))
_track = Track()

post = mk.GaussianState(prior.state_vector, prior.covar, start_time)
for i, measurement in enumerate(measurements):
    _prediction = efk_predictor.predict(post, timestamp=timesteps[i])
    _hypothesis = mk.Hypothesis(prior_state=post, prediction=_prediction, measurement=measurement.state_vector)
    post = efk_updater.update(_hypothesis)
    _track.append(GaussianState(state_vector=post.state_vector, covar=post.covar, timestamp=timesteps[i]))


# %%
# 绘制跟踪结果
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.2)
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements, [0, 2])
plotter.plot_tracks(track, [0, 2], track_label="Stonesoup EKF", marker=dict(symbol="diamond", color="red", size=5))
plotter.plot_tracks(_track, [0, 2], track_label="User EKF", marker=dict(symbol="x", color="green", size=5))
plotter.fig

# %%
""" 保存动画 """
# from drawhelp.io import plotly_fig_to_video_joblib
# plotly_fig_to_video_joblib(plotter.fig, "output_video.mp4", width=1080, height=600)


# %%
""" 打印成C语言数组 """
from myRadar.tool.print_c import numpy_to_c_array

z_list = np.array([m.state_vector[1:4] for m in measurements]).reshape(-1, 3)
c_timestamps = np.array([t.total_seconds() * 1000 for t in (timesteps - start_time)])
x_list = np.array([m.state_vector[0:4] for m in track]).reshape(-1, 4)

lines = [
    "//测量值数量",
    f"static const int N = {len(z_list)};",
    "",
    "//状态向量",
    numpy_to_c_array(x0[:4, 0], "_x0", "double"),
    "",
    "//误差协方差",
    numpy_to_c_array(P0[:4, :4], "_P0", "double"),
    "",
    "//状态转移噪声因子",
    f"static const double _q = {q};",
    "",
    "//测量噪声协方差",
    numpy_to_c_array(R0[1:, 1:], "_R0", "double"),
    "",
    "//时间戳",
    numpy_to_c_array(c_timestamps, "timestamps", "uint32_t"),
    "",
    "//测量值列表",
    numpy_to_c_array(z_list, "_z_list", "double"),
    "",
    "//滤波后状态向量",
    numpy_to_c_array(x_list, "_x_list", "double")
]

# 拼接所有行
output_text = "\n".join(lines)

# 输出：可选择打印到终端或者写入到文件
# 打印到终端
print(output_text)

# 如果需要写入到文件，取消下面代码的注释
# with open("output.data", "w") as f:
#     f.write(output_text)
#%%

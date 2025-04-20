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
from itertools import chain


import plotly.graph_objects as go

import datetime


from core import *


# %%
# 加载数据


file = scipy.io.loadmat(file_name="../data/AT24G_RecordedData_丰海条形模块_单人行走.mat")
radar_data_cube = file["RDM"][1:].transpose(0, 1, 3, 2)

frequency = file["frequency"][0, 0]
bandwidth = file["bandwidth"][0, 0]
numFrame = file["numFrame"][0, 0]
numChannel = file["numChannel"][0, 0]
numSample = file["numSample"][0, 0]
numRangeBin = file["numRangeBin"][0, 0]
numChirp = file["numChirp"][0, 0]
timeChirp = file["timeChirp"][0, 0]
timeChirpGap = file["timeChirpGap"][0, 0]
timeFrameGap = file["timeFrameGap"][0, 0]
del file

timeChirpTotal = timeChirp + timeChirpGap
timeFrame = timeChirpTotal * numChirp
timeFrameTotal = timeFrame + timeFrameGap

resRange = scipy.constants.c / (2 * bandwidth)
resVelocity = scipy.constants.c / (frequency * 2 * timeFrameTotal)

# %%
# 仿真

timestamp_start = datetime.datetime.now()

# 初始化
processor = Processor(
    param=RadarInitParam(
        wavelength=scipy.constants.c / frequency,
        bandwidth=bandwidth,
        rx_antenna_spacing=6.98e-3,
        timeChirp=timeChirp,
        timeChirpGap=timeChirpGap,
        timeFrameGap=timeFrameGap,
        numChannel=numChannel,
        numRangeBin=numRangeBin,
        numChirp=numChirp,
        numMaxCfarPoints=64,
        numMaxCachedFrame=8,
        numInitialMultiMeas=4,
        numInitialCluster=4,
    ),
    config=RadarConfig(
        cfar_cfg=RadarCFARConfig(numTrain=(3, 8), numGuard=(2, 4), thSNR=3.5, thMag=500),
        cfar_filter_cfg=RadarCFARFilterConfig(range0=2, range1=3, shape1=64, th=0.8),
        dbscan_cfg=DBSCANConfig(wr=1, wv=2, eps=0.6, min_samples=5),
        track_cfg=TrackConfig(
            tran_model_q=5,
            meas_noise_r=np.diag([0, 5 / 180 * np.pi, 0.2, 0.07]) ** 2,
            missed_distance=1.5,
            del_unassociated_time=4.0,
            del_missed_probability=0.3,
            init_unassociated_time=2.0,
            init_keep_motion_time=1.5,
            init_keep_static_time=10.0,
            init_speed_th=0.2,
            init_missed_distance=2.0,
            init_covar=np.diag([0.7, 0.2, 0.7, 0.2, 0.7, 0.2]) ** 2,
            fov=np.array([-np.pi * 45 / 180, np.pi * 45 / 180]),
            radius_range=np.array([0.3, 8.0]),
        ),
        channel_phase_diff_threshold=np.pi * 0.9,
    ),
)


from stonesoup.types.state import GaussianState
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.types.detection import Detection

measurement_list = []

all_measurements = []

trajectorys = dict()
unconfirmed_trajectorys = dict()

timestamps = []
for i, rdm in enumerate(radar_data_cube):
    timestamp = timestamp_start + datetime.timedelta(seconds=i * timeFrameTotal)
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
    for target in processor.tracked_targets:
        target: TrackedTarget
        uuid = target.uuid
        if uuid in trajectorys.keys():
            trajectorys[uuid].append(GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp))
        else:
            trajectorys[uuid] = [GaussianState(state_vector=target.state.state_vector, covar=target.state.covar, timestamp=timestamp)]
    for target in processor.unconfirmed_targets:
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
fig

# %%
plotter = AnimatedPlotterly(timestamps, tail_length=0.12)
plotter.plot_measurements(all_measurements, [0, 1], convert_measurements=False)
plotter.plot_tracks(unconfirmed_trajectorys.values(), [0, 2], track_label="起始阶段目标", marker=dict(symbol="x", size=8))
plotter.plot_tracks(trajectorys.values(), [0, 2], track_label="已跟踪目标")

plotter.fig
# %%

go.Figure(data=go.Surface(z=np.sum(np.abs(radar_data_cube[104]), axis=0)))


# %%
# if __name__ == "__main__":
#     from myRadar.plot.io import plotly_fig_to_video_joblib

#     plotly_fig_to_video_joblib(plotter.fig, "output_video.mp4", width=1080, height=600)

# %%

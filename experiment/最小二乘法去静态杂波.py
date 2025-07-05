# %%
import sys

sys.path.append("../")

import numpy as np
from scipy.fft import fftshift, fft, fft2, ifft2, ifft
import scipy.constants
import scipy.io
from dataclasses import dataclass, field


from myRadar.arraysys import angleDualCh
from myRadar.cfar import cfar_2d, cfar_1d

import plotly.graph_objects as go
import drawhelp.draw as dh


# %% 加载数据

def loadmat(file_name):
    global rdm, numFrame, numChannel, numSample, numRangeBin, numChrip, timeChrip, timeChripGap, timeFrameGap, timeFrame, resRange, resVelocity
    file = scipy.io.loadmat(file_name=file_name)
    print(file["INFO"])
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
    del file


loadmat("../data/AT24G_RecordedData_2m处静坐.mat")


# %%
rangefft = ifft(rdm, axis=-1)
ampSpec2DList = np.abs(rdm)

"""  绘制幅度谱 """
dh.draw_2d_spectrumlist(ampSpec2DList[::1, 0, :, :], title="幅度谱")

# %%
""" 提取人体的信号 """
idxHuman = 3

signal_1dfft_human = rangefft[:,:,idxHuman,:]


# %%
""" 绘图观察
"""

signal = signal_1dfft_human[:,0].reshape(-1)

go.Figure(
    data=[
        go.Scatter3d(
            x=np.arange(len(signal)),
            y=np.real(signal),
            z=np.imag(signal),
            mode="lines",
            marker=dict(size=3),
            name="人体多普勒信号",
        )
    ],
    layout=go.Layout(
        title="人体多普勒信号",
        scene=dict(
            aspectmode="manual", aspectratio=dict(x=5, y=1, z=1), xaxis=dict(title="Index"), yaxis=dict(title="Real Part"), zaxis=dict(title="Imaginary Part")
        ),
    ),
)

#%%
"""滑动窗口均值
"""

win_size = 501
start = win_size //2
end = len(signal) - win_size //2

def al_mean(signal, window_size):
    signal = np.asarray(signal)
    result = np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
    # 跳过边缘部分，保留中心部分的均值
    return result  # result 对应 signal[half_k : -half_k]


def fit_circle_least_squares(x, y):
    """
    用最小二乘法拟合圆，返回圆心和半径。
    """
    A = np.column_stack((x, y, np.ones_like(x)))
    Z = x**2 + y**2
    C, *_ = np.linalg.lstsq(A, -Z, rcond=None)
    D, E, F = C
    a, b = -D / 2, -E / 2
    R = np.sqrt(a**2 + b**2 - F)
    return a, b, R

def ls_mean(signal, window_size):
    """
    对复数信号进行滑动窗口最小二乘圆拟合，返回每个窗口拟合得到的圆心序列。
    跳过无法形成完整窗口的边缘部分。
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    signal = np.asarray(signal)
    half_k = window_size // 2
    centers = []

    for i in range(half_k, len(signal) - half_k):
        window = signal[i - half_k : i + half_k + 1]
        x = window.real
        y = window.imag
        a, b, _ = fit_circle_least_squares(x, y)
        centers.append(a + 1j * b)

    return np.array(centers)

mean0 = al_mean(signal, win_size)
mean1 = ls_mean(signal, win_size)

#%%
go.Figure(
    data=[
        go.Scatter3d(
            x=np.arange(len(signal)),
            y=np.real(signal),
            z=np.imag(signal),
            mode="lines",
            marker=dict(size=3),
            name="人体多普勒信号",
        ),
         go.Scatter3d(
            x=np.arange(start, end),
            y=np.real(mean0),
            z=np.imag(mean0),
            mode="lines",
            marker=dict(size=3),
            name="滑动窗口均值",
        ),
           go.Scatter3d(
            x=np.arange(start, end),
            y=np.real(mean1),
            z=np.imag(mean1),
            mode="lines",
            marker=dict(size=3),
            name="最小二乘均值",
        ),
    ],
    layout=go.Layout(
        title="人体多普勒信号",
        scene=dict(
            aspectmode="manual", aspectratio=dict(x=5, y=1, z=1), xaxis=dict(title="Index"), yaxis=dict(title="Real Part"), zaxis=dict(title="Imaginary Part")
        ),
    ),
)

#%%
"""
计算幅度
"""

signal_part = signal[start:end]
mag0 = abs(signal_part - mean0)
mag1 = abs(signal_part - mean1)

print(f"滑动均值去杂波平均幅度：{np.mean(mag0)}\n最小二乘去杂波平均幅度：{np.mean(mag1)}\n提升：{(np.mean(mag1) / np.mean(mag0)-1) * 100:.2f}%")


mag_smooth0 = al_mean(mag0, 500)
mag_smooth1 = al_mean(mag1, 500)

go.Figure(
    data=[
        go.Scatter(
            x=np.arange(len(signal_part)),
            y=mag_smooth0,
            mode="lines",
            name="滑动窗口均值",
        ),
        go.Scatter(
            x=np.arange(len(signal_part)),
            y=mag_smooth1,
            mode="lines",
            name="滑动窗口最小二乘",
        ),
    ]
)

#%%

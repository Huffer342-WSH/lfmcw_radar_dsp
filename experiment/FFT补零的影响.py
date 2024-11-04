# %%
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from pyargus.directionEstimation import DOA_MUSIC
import plotly.graph_objects as go

# 生成一个非整数倍周期的信号
N = 256  # 采样点数
m = 8
f_signal = 50.125  # 信号频率（非整数倍周期）

t = np.arange(N)
signal = np.sin(2 * np.pi / N * f_signal * t)

# %%
# 1. 使用DFT计算频率
ampSpec = np.abs(fft(signal))
idxPeak = np.argmax(ampSpec[: len(ampSpec) // 2])
print(idxPeak)


go.Figure(data=[go.Scatter(x=t, y=ampSpec, mode="lines")]).show()


# 1. 使用DFT计算频率
ampSpec = np.abs(fft(signal, len(signal) * m))
idxPeak = np.argmax(ampSpec[: len(ampSpec) // 2])
print(idxPeak / m)


go.Figure(data=[go.Scatter(x=np.arange(N * m) / m, y=ampSpec, mode="lines")]).show()
# %%

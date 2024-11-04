# %%
import sys

sys.path.append("../")


import numpy as np
import drawhelp.draw as dh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

# %% [markdown]
"""  
室内雷达应用中，需要检测一些慢速运动的目标，这里的慢速是指其速度小于速度分辨率，是得这些目标的信号在RDM中落在速度维度[-1,1]的范围里。
本实验只要研究如何从RDM中小范围的信号中近似的还原慢速运动目标的多普勒信号，得到慢速运动目标的速度

一般来讲，可以逆傅里叶变换+相位解缠绕解决这个问题，算法时间复杂度和空间复杂度都为O(N),N为傅里叶变化的点数，这种方法等同于对多普勒信号做了一个FIR低通滤波然后再分析信号

此处我希望找到一个O(1)的算法，在已知RDM中一个距离单元[r,-1:2]三个复数数据的情况下，推导出慢速目标信号的角度变化（速度）
"""

# %% [markdown]
"""  
##  1. 生成仿真数据
准备一些不同初始相位的，周期相同的信号，先观察相位的影响
"""

# %% 不同相位
N = 64
T = 200
omega = 2 * np.pi / T
t = np.arange(64)
singal_DFTBin1_in_TimeDomain = np.exp(1j * 2 * np.pi / N * t)


def draw8subplot(signal, title=""):
    spec = np.fft.fft(signal, axis=1) / N
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, shared_yaxes=True)
    for i in range(8):
        row = i // 2 + 1  # Determine the row (1 to 4)
        col = i % 2 + 1  # Determine the column (1 or 2)
        fig.add_trace(go.Scatter(x=signal[i].real, y=signal[i].imag, mode="lines"), row=row, col=col)
        fig.add_trace(go.Scatter(x=[0, spec[i][0].real], y=[0, spec[i][0].imag], mode="lines"), row=row, col=col)
        x = singal_DFTBin1_in_TimeDomain * spec[i][1]
        fig.add_trace(go.Scatter(x=x.real, y=x.imag, mode="lines"), row=row, col=col)
    fig.update_layout(
        height=800,
        width=600,
        title_text=title,
        showlegend=False,
        xaxis=dict(scaleanchor="y"),  # Set x and y axes to have 1:1 ratio
        yaxis=dict(scaleanchor="x"),
    )
    fig.update_xaxes(scaleanchor="y", showgrid=True)
    fig.update_yaxes(scaleanchor="x", showgrid=True)
    return fig


initialPhaseList = np.linspace(0, 2 * np.pi, 8)
A = np.linspace(1, 1, 8)
signal = A.reshape([-1, 1]) * np.exp(1j * (omega * t.reshape([1, -1]) + initialPhaseList.reshape(-1, 1)))


fig = draw8subplot(signal, title="不同相位")
fig.show()


# %% 不同幅度
initialPhaseList = np.linspace(0, 2 * np.pi, 8)
A = np.linspace(1, 10, 8)
signal = A.reshape([-1, 1]) * np.exp(1j * (omega * t.reshape([1, -1]) + initialPhaseList.reshape(-1, 1)))


fig = draw8subplot(signal, title="不同幅度")
fig.show()

# %% [markdown]
"""
当输入型号的形状相同时，相位和幅度都不会影响幅度谱的比例

生成不同周期的信号对应的幅度谱的点0和点1的比例
"""

N = 256
M = 10000
T = 1
freqRange = np.linspace(4, 8, M)
t = np.arange(N)
signal = np.exp(1j * 2 * np.pi / N * 9.5 * t)
go.Figure(data=[go.Scatter(x=np.arange(N), y=signal.imag, mode="lines")]).show()
go.Figure(data=[go.Scatter(x=np.arange(N), y=np.abs(np.fft.fft(signal)), mode="lines")]).show()

# %%


def clacRatio_in_FreqDomain(freqRange, N, a, b):
    freq = np.linspace(freqRange[0], freqRange[1], M)
    signal = np.exp(1j * 2 * np.pi / N * freq.reshape(-1, 1) * np.arange(N))
    print(signal.shape)
    ampSpec = np.abs(np.fft.fft(signal, axis=-1))
    ratio = ampSpec[:, a] / ampSpec[:, b]
    return ratio, freq


ratio, freq = clacRatio_in_FreqDomain([30, 31], N, 30, 31)

go.Figure(
    data=[go.Scatter(x=freq, y=ratio, mode="lines")],
    layout={"xaxis": {"title": "比例"}, "yaxis": {"title": "角度 (单位:pi)"}},
).show()
# %%
"""拟合多项式曲线"""


# 定义分式函数
def rational_function(x, a, b, c, d):
    return (a * x + b) / (c * x + d)


def rational_function2(x):
    return 2 * np.pi * (x / (x + 1))


# 使用 curve_fit 进行拟合
params, _ = curve_fit(rational_function, ratio, freq)
params = params / params[3]
# params = np.around(32767 / np.max(np.abs(params)) * params).astype(np.int16)
y_values = rational_function(ratio, *params)
y_values2 = rational_function2(ratio)
go.Figure(
    data=[
        go.Scatter(x=ratio, y=freq, mode="lines", name="原曲线"),
        go.Scatter(x=ratio, y=y_values, mode="lines", name="拟合后曲线"),
    ],
    layout={"title": "拟合曲线", "xaxis_title": r"$y=\frac{ax+b}{cx+d}$"},
).show()
rmse = np.sqrt(np.mean((y_values - freq) ** 2))
print("公式参数为：", params)
print("均方根误差为：", rmse)


# %% [markdown]


# %%

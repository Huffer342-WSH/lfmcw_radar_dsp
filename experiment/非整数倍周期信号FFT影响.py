# %%
import sys

sys.path.append("../")


import numpy as np
import drawhelp.draw as dh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit


# %% [markdown]
# # 简介
#
# 室内雷达应用中，需要检测一些慢速运动的目标，这里的慢速是指其速度小于速度分辨率，是得这些目标的信号在RDM中落在速度维度[-1,1]的范围里。
#
# 本实验只要研究如何从RDM中小范围的信号中近似的还原慢速运动目标的多普勒信号，得到慢速运动目标的速度
#
# 一般来讲，可以逆傅里叶变换+相位解缠绕解决这个问题，算法时间复杂度和空间复杂度都为O(N),N为傅里叶变化的点数，这种方法等同于对多普勒信号做了一个FIR低通滤波然后再分析信号
#
# 此处我希望找到一个O(1)的算法，在已知RDM中一个距离单元[r,-1:2]三个复数数据的情况下，推导出慢速目标信号的角度变化（速度）
#
#
# update:2025/7/5
# 该实验拓展为针对：信号并不在FFT分辨率落点上时，通过拟合的方式提高精度

# %% [markdown]
# ##  1. 生成仿真数据
#
# 准备一些不同初始相位的，周期相同的信号，先观察相位的影响

# %%
N = 256
T = 1
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


# %%
initialPhaseList = np.linspace(0, 2 * np.pi, 8)
A = np.linspace(1, 10, 8)
signal = A.reshape([-1, 1]) * np.exp(1j * (omega * t.reshape([1, -1]) + initialPhaseList.reshape(-1, 1)))


fig = draw8subplot(signal, title="不同幅度")
fig.show()


# %% [markdown]
# 当输入型号的形状相同时，相位和幅度都不会影响幅度谱的比例
#
#
#
# 生成不同周期的信号对应的幅度谱的点0和点1的比例

# %%
M = 10000


def clacRatio_in_FreqDomain(freqRange, N, a, b):
    freq = np.linspace(freqRange[0], freqRange[1], M)
    signal = np.exp(1j * 2 * np.pi / N * freq.reshape(-1, 1) * np.arange(N))
    print(signal.shape)
    ampSpec = np.abs(np.fft.fft(signal, axis=-1))
    ratio = ampSpec[:, a] / ampSpec[:, b]
    return ratio, freq


ratio, freq = clacRatio_in_FreqDomain([0.5, 1], N, 1, 0)

go.Figure(
    data=[go.Scatter(x=freq, y=ratio, mode="lines")],
    layout={"xaxis": {"title": "比例"}, "yaxis": {"title": "角度 (单位:pi)"}},
).show()

# %%
"""拟合多项式曲线"""

# 定义待拟合曲线格式
def rational_function(x, a, b, c, d):
    return (a * x + b) / (c * x + d)

# 使用 curve_fit 进行拟合
params, _ = curve_fit(rational_function, ratio, freq)
params = params / max(params)
print("公式参数为：", params)


# 根据拟合出来的参数定义函数
def rational_function2(x):
    return 2 * np.pi * (x / (x + 1))

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
print("均方根误差为：", rmse)


# %%
curve_fit(rational_function, ratio, freq)

# %% [markdown]
# ## 数学模型
#
# ### 信号定义
#
# 考虑归一化频率为 $ f $ 的复指数信号：
#
# $$
# x[n] = e^{j2\pi f n}, \quad n=0,1,\cdots,N-1
# $$
#
# 其DFT定义为：
#
# $$
# X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}
# $$
#
# ### 等比数列求和
#
# 将信号表达式代入DFT：
#
# $$
# X[k] = \sum_{n=0}^{N-1} e^{j2\pi f n} \cdot e^{-j2\pi kn/N} = \sum_{n=0}^{N-1} e^{j2\pi n (f - k/N)}
# $$
#
# 这是一个首项为 $ a_1 = 1 $，公比为 $ r = e^{j2\pi (f - k/N)} $ 的等比数列。等比数列求和公式为：
#
# $$
# S_N = a_1 \frac{1 - r^N}{1 - r}
# $$
#
# 代入参数：
#
# $$
# X[k] = \frac{1 - [e^{j2\pi (f - k/N)}]^N}{1 - e^{j2\pi (f - k/N)}}
# $$
#
# ### 简化表达式
#
# 化简分子：
#
# $$
# [e^{j2\pi (f - k/N)}]^N = e^{j2\pi N(f - k/N)} = e^{j2\pi (Nf - k)}
# $$
#
# 由于 $ k $ 是整数，$ e^{-j2\pi k} = 1 $，所以：
#
# $$
# e^{j2\pi (Nf - k)} = e^{j2\pi Nf}
# $$
#
# 因此分子简化为：
#
# $$
# 1 - e^{j2\pi Nf}
# $$
#
# 最终DFT表达式为：
#
# $$
# X[k] = \frac{1 - e^{j2\pi Nf}}{1 - e^{j2\pi (f - k/N)}}
# $$
#
# ### 幅度谱表达式
#
# 幅度谱为：
#
# $$
# |X[k]| = \left| \frac{1 - e^{j2\pi Nf}}{1 - e^{j2\pi (f - k/N)}} \right|
# $$
#
# 利用复指数性质 $|1 - e^{j\theta}| = 2|\sin(\theta/2)|$：
#
# $$
# |X[k]| = \frac{|\sin(\pi Nf)|}{|\sin[\pi (f - k/N)]|}
# $$
#
#
# ## 任意两点幅度比值计算
#
# $|X[k]|$表达式的分子是一个不随序号$k$改变的值，那么通过相除可以去掉该项
#
# 考虑频点 $a$ 和 $b$，其幅度比为：
#
# $$
# \frac{|X[a]|}{|X[b]|} = \frac{|\sin[\pi (f - a/N)]|^{-1}}{|\sin[\pi (f - b/N)]|^{-1}} = \frac{|\sin[\pi (f - b/N)]|}{|\sin[\pi (f - a/N)]|}
# $$
#
#
# 当 $f-b/N$ 的值非常小时，可以应用近似：
#
# $$
# sin(x) = x
# $$
#
# 因此我们取点峰值点$k_1$与相邻点$k_1 \pm 1$计算比值
#
# 在峰值邻域（$k_1$及其相邻点），设$f = \frac{k_1 + \delta}{N}$，则：
#
# 1. 峰值点$k_1$：
# $$
#    |X[k_1]| \propto \frac{1}{|\sin(\pi \delta/N)|}
# $$
#
# 2. 相邻点$k_1 \pm 1$：
# $$
#    |X[k_1 \pm 1]| \propto \frac{1}{|\sin[\pi (\delta \mp 1)/N]|}
# $$
#
# ### 情况1：$\delta \geq 0$(取右侧点$k_1+1$)
# $$
# r = \frac{|X[k_1]|}{|X[k_1+1]|} = \frac{|\sin[\pi (\delta - 1)/N]|}{|\sin(\pi \delta/N)|}
# $$
#
# 当$N$较大时，小角度近似$\sin\theta \approx \theta$：
# $$
# r \approx \frac{|\pi (\delta - 1)/N|}{|\pi \delta/N|} = \frac{|1 - \delta|}{\delta} = \frac{1 - \delta}{\delta}
# $$
#
# 解得：
# $$
# \delta \approx \frac{1}{r + 1}
# $$
#
# ### 情况2：$\delta < 0$(取左侧点$k_1-1$)
# $$
# r = \frac{|X[k_1]|}{|X[k_1-1]|} = \frac{|\sin[\pi (\delta + 1)/N]|}{|\sin(\pi \delta/N)|} \approx \frac{|1 + \delta|}{|\delta|} = -\frac{1 + \delta}{\delta}
# $$
#
#
# 解得：
# $$
# \delta \approx \frac{-1}{r + 1}
# $$
#
# ### 统一表达式
# 两种情况的解可统一表示为：
# $$
# \delta \approx \frac{b}{r + 1}
# $$
# 其中：
# $$
# b = \begin{cases}
# 1 & \delta \geq 0 \\
# -1 & \delta < 0
# \end{cases}
# $$
#
# ### 综上
#
# 我们计算拟合频点的方法为
# 1. 找到幅度谱$|X|$ 的峰值点 $k_1$ 以及相邻点中较大的点 $k_1 \pm 1$
# 2. 计算 $r = \frac{|X[k_1]|}{|X[k_1 \pm 1]|}$， b = $1$ or $-1$
# 3. 根据 $r$ DFT频点 $fN$
#     $$
#     fN = k_1 + \delta = k_1 + \frac{b}{r + 1} ,
#
#     b = \begin{cases}
#     1 & \delta \geq 0 \\
#     -1 & \delta < 0
#     \end{cases}
#     $$
#
#
# ## 其他
#
# 加入选择峰值点和较小的相邻点推导，公式为
#
# $$
# fN =  k_1 + \frac{b}{1-r} ,
#
# b = \begin{cases}
# 1 & \delta \geq 0 \\
# -1 & \delta < 0
# \end{cases}
# $$

# %% [markdown]
# ## 插值算法
#
# 拓展上文可以得到两种插值方式：
# 1. 使用峰值点和幅度较大的相邻点
# 2. 使用峰值点和幅度较小的相邻点
#
# 第二种方法精度更差，但是抗干扰能力好，具体见下文


# %%
def fit_func1(x0, x1, x2, y0, y1, y2):
    """
    上文参数得到的拟合函数，输入数据为峰值点附近三个点,取峰值点和相邻点的较大点
    """
    a = x1
    b = -1 if (y0 > y2) else 1
    r = y1 / (y0 if (y0 > y2) else y2)
    x_peak = a + b / (r + 1)
    return x_peak


def fit_func2(x0, x1, x2, y0, y1, y2):
    """
    上文参数得到的拟合函数，输入数据为峰值点附近三个点,取峰值点和相邻点的较小点
    """
    a = x1
    b = -1 if (y0 > y2) else 1
    r = y1 / (y2 if (y0 > y2) else y0)
    # if (r > 1.5):
    #     x_peak = a + b / (r - 1)
    # else:
    #     x_peak = fit_func1(x0, x1, x2, y0, y1, y2)
    x_peak = a + b / (r - 1)
    return x_peak


def fit_func3(x0, x1, x2, y0, y1, y2):
    """
    3点拟合二次曲线找峰值点，输入数据为峰值点附近三个点
    """
    X = np.array(
        [
            [x0**2, x0, 1],
            [x1**2, x1, 1],
            [x2**2, x2, 1],
        ]
    )
    Y = np.array([y0, y1, y2])

    a, b, c = np.linalg.solve(X, Y)

    if a == 0:
        return None
    x_peak = -b / (2 * a)
    return x_peak


def genAmpSpec(omega, N):
    signal = np.exp(1j * 2 * np.pi / N * omega * np.arange(N))
    ampSpec = np.abs(np.fft.fft(signal))
    return ampSpec


def findPeak3(ampSpec):
    idxPeak = np.argmax(ampSpec)
    idx0 = max(idxPeak - 1, 0)
    idx1 = idxPeak
    idx2 = min(idxPeak + 1, len(ampSpec) - 1)
    return idx0, idx1, idx2, ampSpec[idx0], ampSpec[idx1], ampSpec[idx2]


# %% [markdown]
# ## 测试
#
# ### 单目标测试
#
# 准备一个测试函数，生成单目标幅度谱分别使用三种插值方法计算，绘图比较结果


# %%
def test(x):
    """
    测试函数，用于生成信号的幅度谱并寻找峰值位置

    参数:
    x -- 输入频率

    返回:
    res1 -- 使用自定义拟合函数1计算的峰值位置
    res2 -- 使用二次拟合函数计算的峰值位置
    """
    ampSpec = genAmpSpec(x, N)
    idxPeak = findPeak3(ampSpec)
    res1 = fit_func1(*idxPeak)
    res2 = fit_func2(*idxPeak)
    res3 = fit_func3(*idxPeak)

    return res1, res2, res3


xs = np.arange(2, 10, 0.02)
ys_list = [test(x) for x in xs]
y1 = [ys[0] for ys in ys_list]
y2 = [ys[1] for ys in ys_list]
y3 = [ys[2] for ys in ys_list]

go.Figure(
    data=[
        go.Scatter(x=xs, y=y1 - xs, mode="lines", name="fft专用拟合1"),
        go.Scatter(x=xs, y=y2 - xs, mode="lines", name="fft专用拟合2"),
        go.Scatter(x=xs, y=y3 - xs, mode="lines", name="二次拟合"),
    ],
    layout={"title": "拟合误差", "xaxis_title": "频率"},
)

# %% [markdown]
# ### 相邻目标测试
#
# 设置两个相邻的目标, 这些拟合方法都会因为第二个目标的干扰导致结果想令一个目标靠近

# %%
ampSpec = genAmpSpec(9, N) + genAmpSpec(10, N)
go.Figure(data=[go.Scatter(x=np.arange(N), y=ampSpec, mode="lines")]).show()

res1 = (
    fit_func1(8, 9, 10, ampSpec[8], ampSpec[9], ampSpec[10]),
    fit_func1(9, 10, 11, ampSpec[9], ampSpec[10], ampSpec[11]),
)
res2 = (
    fit_func2(8, 9, 10, ampSpec[8], ampSpec[9], ampSpec[10]),
    fit_func2(9, 10, 11, ampSpec[9], ampSpec[10], ampSpec[11]),
)
res3 = (
    fit_func3(8, 9, 10, ampSpec[8], ampSpec[9], ampSpec[10]),
    fit_func3(9, 10, 11, ampSpec[9], ampSpec[10], ampSpec[11]),
)

print(f"")
print(f"res1= {res1}")
print(f"res2= {res2}")
print(f"res3= {res3}")

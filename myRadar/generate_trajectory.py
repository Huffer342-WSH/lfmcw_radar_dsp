# %%
import numpy as np
import plotly.graph_objects as go
import drawhelp.draw as dh


def trajGen_line(posStart: np.ndarray, velocity: np.ndarray, freqSampling, numSamples):
    # 初始化数组来存储时间和位置
    times = np.arange(0, numSamples) / freqSampling
    positions = np.matmul(times.reshape((-1, 1)), velocity.reshape((1, 3))) + posStart.reshape((1, 3))

    return times, positions


# %%
# 示例输入
if __name__ == "__main__":

    initial_position = np.array([10, 0, 0])  # 初始位置 (x, y, z)
    velocity = np.array([10, 5, 2])  # 速度矢量 (x, y, z), 单位为 m/s
    sampling_frequency = 100  # 采样频率 (单位：Hz)
    num_samples = 100  # 采样点数

    # 生成轨迹
    times, positions = trajGen_line(initial_position, velocity, sampling_frequency, num_samples)

    # 输出时刻和位置
    plotData = [[go.Scatter(x=[pos[0]], y=[pos[1]], mode="markers")] for pos in positions]
    fig = dh.draw_animation(listData=plotData)
    fig.update_layout(xaxis_range=(0, 40), yaxis_range=(-10, 30))
    fig.show()

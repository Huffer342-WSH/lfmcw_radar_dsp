#%% [markdown]
# ###  1. 圆的方程
#
# 圆的一般形式为：
#
# $$
# (x - a)^2 + (y - b)^2 = R^2
# $$
#
# 展开为：
#
# $$
# x^2 + y^2 - 2a x - 2b y + a^2 + b^2 - R^2 = 0
# $$
#
# 记 $D = -2a$，$E = -2b$，$F = a^2 + b^2 - R^2$，可以将其变形为线性形式：
#
# $$
# x^2 + y^2 + Dx + Ey + F = 0
# $$
#
# ---
#
# ### 2. 最小二乘法思路
#
# 已知 $n$ 个点 $(x_i, y_i)$，令：
#
# $$
# z_i = x_i^2 + y_i^2
# $$
#
# 我们要最小化下面这个残差平方和：
#
# $$
# \sum_{i=1}^n (z_i + D x_i + E y_i + F)^2
# $$
#
# 这实际上是一个典型的线性最小二乘问题，我们可以设：
#
# * 矩阵 $A = \begin{bmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ \vdots & \vdots & \vdots \\ x_n & y_n & 1 \end{bmatrix}$
# * 向量 $Z = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_n \end{bmatrix}$
#
# 然后解线性最小二乘问题：
#
# $$
# A \cdot \begin{bmatrix} D \\ E \\ F \end{bmatrix} = -Z
# $$
#
# ---
#
# ### 3. 计算圆心和半径
#
# 一旦解出 $D, E, F$，我们可以恢复：
#
# $$
# a = -\frac{D}{2},\quad b = -\frac{E}{2},\quad R = \sqrt{a^2 + b^2 - F}
# $$
#

#%%

import numpy as np
import plotly.graph_objects as go



def generate_noisy_circle_points(center, radius, noise_std=0.1, num_points=100, angle_range=(0, 2*np.pi), seed=None):
    """
    生成围绕给定圆心和半径的带高斯噪声的圆弧上的点。

    Parameters:
        center (tuple): 圆心 (a, b)
        radius (float): 半径
        noise_std (float): 噪声标准差（高斯分布）
        num_points (int): 点的数量
        angle_range (tuple): 角度范围 (theta_start, theta_end)，单位为弧度
        seed (int): 随机种子，确保可重复性

    Returns:
        x, y: numpy数组，表示生成的点坐标
    """
    if seed is not None:
        np.random.seed(seed)

    theta_start, theta_end = angle_range
    angles = np.linspace(theta_start, theta_end, num_points)

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    x_noisy = x + np.random.normal(0, noise_std, size=x.shape)
    y_noisy = y + np.random.normal(0, noise_std, size=y.shape)

    return x_noisy, y_noisy



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


def plot_circles(x, y, true_center, true_radius, fit_center, fit_radius):
    """
    使用 plotly 绘制原始点，真实圆和拟合圆。
    """
    theta = np.linspace(0, 2*np.pi, 300)
    true_x = true_center[0] + true_radius * np.cos(theta)
    true_y = true_center[1] + true_radius * np.sin(theta)

    fit_x = fit_center[0] + fit_radius * np.cos(theta)
    fit_y = fit_center[1] + fit_radius * np.sin(theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Noisy Points'))
    fig.add_trace(go.Scatter(x=true_x, y=true_y, mode='lines', name='True Circle', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', name='Fitted Circle'))

    fig.update_layout(
        title='最小二乘法拟合圆',
        xaxis_title='X',
        yaxis_title='Y',
        width=600,
        height=600,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()


#%%
# 参数
true_center = (2.0, -1.5)
true_radius = 5.0
noise_std = 0.2
num_points = 20

# 生成点
x, y = generate_noisy_circle_points(center=true_center, radius=5,noise_std=noise_std,num_points=num_points, angle_range=(0, 0.6*np.pi))

# 拟合
fit_center = fit_circle_least_squares(x, y)
print(f"真实圆心: {true_center}, 半径: {true_radius}")
print(f"拟合圆心: {fit_center[:2]}, 半径: {fit_center[2]}")

# 画图
plot_circles(x, y, true_center, true_radius, fit_center[:2], fit_center[2])

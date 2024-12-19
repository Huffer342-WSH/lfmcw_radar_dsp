import numpy as np
import plotly.graph_objects as go


def genScatterPolar(coordinates, tail_length: int = 10, name: str = None):
    """
    生成极坐标散点图的函数。

    参数
    ----------
    coordinates : list of list of tuples
        每个内层列表包含一组坐标，坐标以 (角度, 半径) 的元组形式表示。
    tail_length : int, optional
        每个散点图中显示的点的最大数量，默认为10。
    name : str, optional
        每个散点图的名称，默认为None。

    返回
    -------
    list of plotly.graph_objects.Scatter
        每个散点图对象表示一帧的可视化数据。
    """

    num_frame = len(coordinates)
    num_point = 0
    for c in coordinates:
        num_point += len(c)

    x = np.zeros(num_point)
    y = np.zeros(num_point)
    count = 0
    idx = np.zeros(num_frame + 1).astype(int)
    for i, c in enumerate(coordinates):
        for j in c:
            x[count] = j[1] * np.cos(j[0])
            y[count] = j[1] * np.sin(j[0])
            count += 1
        idx[i + 1] = count

    ret = []
    for end in range(num_frame):
        start = max(0, end - tail_length)
        ret.append(go.Scatter(x=x[idx[start] : idx[end + 1]], y=y[idx[start] : idx[end + 1]], mode="markers", name=name))
    return ret

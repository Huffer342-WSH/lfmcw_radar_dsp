import numpy as np


def polar2cart(polar_coords):
    """
    将极坐标转换为直角坐标

    Parameters
    ----------
    polar_coords : (N, 2) or (N, 3) array_like
        极坐标矩阵

    Returns
    -------
    (N, 2) or (N, 3) array_like
        直角坐标矩阵
    """
    polar_coords = np.asarray(polar_coords)

    # 如果输入是2D极坐标 (r, theta) 或一个点 (2,)
    if polar_coords.shape[-1] == 2:
        return np.column_stack((polar_coords[:, 0] * np.cos(polar_coords[:, 1]), polar_coords[:, 0] * np.sin(polar_coords[:, 1])))

    # 如果输入是3D极坐标 (r, theta, phi) 或一个点 (3,)
    elif polar_coords.shape[-1] == 3:
        r, theta, phi = polar_coords.T
        return np.column_stack((r * np.cos(theta) * np.cos(phi), r * np.sin(theta) * np.cos(phi), r * np.sin(phi)))

    else:
        raise ValueError("输入的极坐标必须是形状为 (N, 2) 或 (N, 3) 的矩阵，或形状为 (2,) 或 (3,) 的数组。")

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

    # 处理二维极坐标 (N, 2) 或 (2,)
    if polar_coords.shape[-1] == 2:
        r = polar_coords[..., 0]
        theta = polar_coords[..., 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=-1)

    # 处理三维极坐标 (N, 3) 或 (3,)
    elif polar_coords.shape[-1] == 3:
        r = polar_coords[..., 0]
        theta = polar_coords[..., 1]  # 方位角
        phi = polar_coords[..., 2]  # 俯仰角
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        return np.stack([x, y, z], axis=-1)

    else:
        raise ValueError("输入的极坐标数组形状不正确，应该是 (N, 2), (N, 3), (2,) 或 (3,)")

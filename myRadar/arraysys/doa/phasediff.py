import numpy as np


def angleDualCh(complex0, complex1):
    """
    使用相位差法计算角度，支持数组输入

    Args:
        complex0: 第一个通道的复数信号，可以是单个复数或复数数组
        complex1: 第二个通道的复数信号，可以是单个复数或复数数组

    Returns:
        theta: 角度，弧度，数组与输入的形状一致
    """
    # 计算复数信号的相位
    phase0 = np.angle(complex0)
    phase1 = np.angle(complex1)

    # 计算相位差
    phaseDelta = phase0 - phase1

    # 使用 np.where 处理相位差的边界
    phaseDelta = np.where(phaseDelta < -np.pi, phaseDelta + 2 * np.pi, phaseDelta)
    phaseDelta = np.where(phaseDelta > np.pi, phaseDelta - 2 * np.pi, phaseDelta)

    # 计算角度
    theta = np.arcsin(phaseDelta / np.pi)

    return theta

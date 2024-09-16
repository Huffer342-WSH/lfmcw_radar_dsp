# %%

import numpy as np
import scipy.interpolate
import scipy.constants
import joblib


def generateRadarDataCube(frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo):
    """
    生成LFMCW雷达数据立方体

    Parameters
    ----------
    frequency : float
        载波频率
    bandwidth : float
        带宽
    timeChrip : float
        一个chrip的持续时间
    timeIdle : float
        两个chrip之间的时间间隔
    timeNop : float
        一帧数据的结尾
    freqSampling : float
        采样频率
    numSampling : int
        采样点数
    numChrip : int
        chrip个数
    numFrame : int
        帧数
    posTx : array_like
        发射天线的位置，shape=(N,3)，N为发射天线的个数，每一行表示一个天线的坐标
    posRx : array_like
        接收天线的位置，格式posTx
    targetsInfo : list
        目标的信息

    Returns
    -------
    signal : array_like
        LFMCW雷达数据立方体，4-D数组，四个维度分别为(帧序号,通道序号,chrip序号,采样点序号)

    Examples
    --------

    """
    numTx = len(posTx)
    numRx = len(posRx)

    unusedChrip = timeChrip - (numSampling - 1) / freqSampling
    axisTime = (
        np.tile(np.linspace(0, numSampling / freqSampling, numSampling, endpoint=False), numChrip * numFrame)
        + np.repeat(np.linspace(0, numChrip * numFrame * (timeIdle + timeChrip), numChrip * numFrame, endpoint=False), numSampling)
        + np.repeat(np.linspace(0, numFrame * timeNop, numFrame, endpoint=False), numSampling * numChrip)
    ) + unusedChrip / 2

    def clacPhase(axisT, fc, timeChrip, timeIdle, timeNop, numChrip, slope):
        t = (axisT % ((timeChrip + timeIdle) * numChrip + timeNop)) % (timeChrip + timeIdle)
        phase = 2 * np.pi * t * (fc + 0.5 * slope * t)
        return phase

    def clacPhase_parallel(axisT, fc, timeChrip, timeIdle, timeNop, numChrip, slope):
        num_slices = joblib.cpu_count() * 8
        axisT_slices = np.array_split(axisT, num_slices)
        results = joblib.Parallel(n_jobs=-1)(joblib.delayed(clacPhase)(slice_, fc, timeChrip, timeIdle, timeNop, numChrip, slope) for slice_ in axisT_slices)
        return np.concatenate(results)

    phaseTx = clacPhase_parallel(axisTime, frequency, timeChrip, timeIdle, timeNop, numChrip, bandwidth / timeChrip)

    signal = np.zeros((numTx * numRx, numSampling * numChrip * numFrame), dtype=np.complex128)

    tragetsPos = []
    for target in targetsInfo:
        # 1. 补全目标的位置信息，对于目标轨迹时间轴之外的时间点，使用外插法补全
        # 使用原始时间轴的边界值进行填充
        interp_pos = scipy.interpolate.interp1d(
            target["times"], target["pos"], axis=0, kind="quadratic", bounds_error=False, fill_value=(target["pos"][0], target["pos"][-1])
        )

        posNew = interp_pos(axisTime)
        tragetsPos.append(posNew)
        # 2. 计算目标在不同时刻时的 电磁波从发射然后经过反射最后被天线接收的时间延迟，减去延迟和得到新的时间轴

        for i in range(numTx):
            for j in range(numRx):
                index = i * numRx + j
                tmpTx = posTx[i]
                tmpRx = posRx[j]
                disTx = np.linalg.norm(posNew - tmpTx, axis=1)
                disRx = np.linalg.norm(posNew - tmpRx, axis=1)
                txTimeStamp = axisTime - (disTx + disRx) / scipy.constants.c

                signal[index, :] += (
                    target["rsc"]
                    / (disTx**2)
                    / (disRx**2)
                    * np.exp(1j * (phaseTx - clacPhase_parallel(txTimeStamp, frequency, timeChrip, timeIdle, timeNop, numChrip, bandwidth / timeChrip)))
                )
    tragetsPos = np.array(tragetsPos)
    signal = signal.reshape(numTx * numRx, numFrame, numChrip, numSampling).swapaxes(0, 1)

    return signal, tragetsPos


# %% 测试
if __name__ == "__main__":
    import cProfile
    import drawhelp.draw as dh
    import plotly.graph_objects as go
    from myRadar.generate_trajectory import trajGen_line
    from scipy.fft import fft, fftshift, ifft, fft2

    frequency = 24e9
    bandwidth = 1000e6
    timeChrip = 150e-6
    timeIdle = 200e-6
    timeNop = 2000e-6  #
    freqSampling = 1e6

    numSampling = 128
    numChrip = 32
    numFrame = 10

    posTx = np.array([[0, 0, 0]])
    posRx = np.array([[0, -0.25 * scipy.constants.c / frequency, 0], [0, 0.25 * scipy.constants.c / frequency, 0]])

    # targetsInfo包含多个目标的轨迹，每个目标用一个字典储存。

    times, positions = trajGen_line(np.array([10, 5, 0]), np.array([-1, 0, 0]), freqSampling, freqSampling * 10)

    target0 = dict(rsc=1, times=times, pos=positions)

    times, positions = trajGen_line(np.array([5, 0, 0]), np.array([1, 0, 0]), freqSampling, freqSampling * 10)
    target1 = dict(rsc=1, times=times, pos=positions)

    targetsInfo = [target0, target1]

    signal, _ = generateRadarDataCube(
        frequency, bandwidth, timeChrip, timeIdle, timeNop, freqSampling, numSampling, numChrip, numFrame, posTx, posRx, targetsInfo
    )

    frame0 = signal[0, 0, :]
    frame1 = signal[0, 1, :]

    # 观察RDM
    resRange = scipy.constants.c / (2 * bandwidth * (numSampling / freqSampling) / timeChrip)
    axis_x = np.arange(0, 128) * resRange
    axis_y = np.arange(-16, 16) * scipy.constants.c / (2 * frequency * (timeIdle + timeChrip) * 31)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(frame0), axes=0))
    # dh.draw_spectrum(spec, x=axis_x, y=axis_y)
    dh.draw_spectrum(spec)

    # 相位法测角

    pos0 = (14, 64)
    pos1 = (18, 29)

    rdm0 = fftshift(fft2(frame0), axes=0)
    rdm1 = fftshift(fft2(frame1), axes=0)

    cplx0 = rdm0[pos0[0], pos0[1]]
    cplx1 = rdm1[pos0[0], pos0[1]]
    phsaeDelta = np.arctan2(cplx0.imag, cplx0.real) - np.arctan2(cplx1.imag, cplx1.real)
    if phsaeDelta < -np.pi:
        phsaeDelta += 2 * np.pi
    elif phsaeDelta > np.pi:
        phsaeDelta -= 2 * np.pi
    theta = np.arcsin(phsaeDelta / np.pi)
    print(pos0[1] * resRange * np.cos(theta), pos0[1] * resRange * np.sin(theta))


# %%

import numpy as np
from scipy.signal import convolve2d, convolve
from myRadar.basis import PrefixSum2D

__all__ = ["cfar_2d", "cfar_1d"]


def cfar_2d(mat, numTrain, numGuard, threshold, type="CrossMean"):
    numTrain = np.array(numTrain)
    numGuard = np.array(numGuard)

    shape = np.array([1, 1]) + 2 * numTrain + 2 * numGuard
    convKernel = np.zeros(shape)

    # CA-CFAR,十字形状
    if type == "CA" or type == "Cell-Averanging":
        convKernel[: numTrain[0], np.floor_divide(shape[1], 2)] = 1
        convKernel[-numTrain[0] :, np.floor_divide(shape[1], 2)] = 1
        convKernel[np.floor_divide(shape[0], 2), : numTrain[1]] = 1
        convKernel[np.floor_divide(shape[0], 2), -numTrain[1] :] = 1
        convKernel /= np.sum(convKernel)
        noise_level = convolve2d(mat, convKernel, mode="same", boundary="wrap")
        coords = np.argwhere(mat / noise_level > threshold)
    elif type == "GOCA" or type == "Greatest-of-Cell-Averaging":
        coords, noise_level = cfar2d_goca(mat, numTrain, numGuard, threshold)
    else:
        raise NotImplementedError("unKnown CFAR type.")

    return coords, noise_level


def cfar_2d_cross_prefix(mat, numTrain, numGuard, threshold):
    """
    Optimized 2D CFAR with cross-shaped training and guard cells using prefix sum for fast area sum calculation.

    Parameters:
    - mat: 2D input array
    - numTrain: list or tuple with two elements, specifying the number of training cells along each axis.
    - numGuard: list or tuple with two elements, specifying the number of guard cells along each axis.
    - threshold: float, threshold factor to apply to the noise level.

    Returns:
    - cfar_output: 2D output array of the same size as input.
    """
    rows, cols = mat.shape
    numTrainX, numTrainY = numTrain
    numGuardX, numGuardY = numGuard

    # Pad the matrix using wrap mode
    padded_mat = np.pad(mat, ((numTrainX + numGuardX, numTrainX + numGuardX), (numTrainY + numGuardY, numTrainY + numGuardY)), mode="wrap")

    # Compute the prefix sum matrix using numpy.cumsum with an extra row and column of zeros
    ps = PrefixSum2D(padded_mat)

    noise_level = np.zeros_like(mat, dtype=np.float64)

    # Predefine relative coordinates for the four areas (left, right, up, down)
    areas = np.array(
        [
            [[-numTrainX - numGuardX, 0], [-numGuardX - 1, 0]],  # 上
            [[numGuardX + 1, 0], [numTrainX + numGuardX, 0]],  # 下
            [[0, -numTrainY - numGuardY], [0, -numGuardY - 1]],  # 左
            [[0, numGuardY + 1], [0, numTrainY + numGuardY]],  # 右
        ]
    )

    # Iterate over every element in the matrix
    for i in range(rows):
        for j in range(cols):
            means = []

            # Loop through the four areas (left, right, up, down)
            for area in areas:
                center = np.array([i, j]) + np.array([numTrainX + numGuardX, numTrainY + numGuardY])
                num_cells = (area[1, 0] - area[0, 0] + 1) * (area[1, 1] - area[0, 1] + 1)
                if num_cells == 0:
                    means.append(0)
                else:
                    a = area + center
                    area_sum = ps.getSum(a[0, 0], a[0, 1], a[1, 0], a[1, 1])
                    means.append(area_sum / num_cells)

            # Find the maximum noise level from the four directions
            noise = max(means)

            # Apply thresholding

            noise_level[i, j] = noise
    coords = np.argwhere(mat / noise_level > threshold)

    return coords, noise_level


def cfar2d_goca(mat, numTrain, numGuard, threshold):
    padWidth = np.array([numTrain[0] + numGuard[0], numTrain[1] + numGuard[1]])
    padded_mat = np.pad(
        mat,
        pad_width=((numTrain[0] + numGuard[0], numTrain[0] + numGuard[0]), (numTrain[1] + numGuard[1], numTrain[1] + numGuard[1])),
        mode="wrap",
    )
    trainMean = np.zeros(shape=(4, mat.shape[0], mat.shape[1]))  # 多加一个维度用来保存四个分支的均值

    # 计算维度0方向上训练单元的和
    prefixSum0 = np.zeros(shape=(mat.shape[0] + 1 + padWidth[0] * 2, mat.shape[1]))
    prefixSum0[1:, :] = np.cumsum(padded_mat[:, padWidth[1] : -padWidth[1]], axis=0)
    numCell = numTrain[0]
    trainSum0_full = (prefixSum0[numCell:, :] - prefixSum0[:-numCell, :]) / numCell
    trainMean[0] = trainSum0_full[: mat.shape[0], :]
    trainMean[1] = trainSum0_full[-mat.shape[0] :, :]

    # 计算维度1方向上训练单元的和
    prefixSum1 = np.zeros(shape=(mat.shape[0], mat.shape[1] + 1 + padWidth[1] * 2))
    prefixSum1[:, 1:] = np.cumsum(padded_mat[padWidth[0] : -padWidth[0], :], axis=1)
    numCell = numTrain[1]
    trainSum1_full = (prefixSum1[:, numCell:] - prefixSum1[:, :-numCell]) / numCell
    trainMean[2] = trainSum1_full[:, : mat.shape[1]]
    trainMean[3] = trainSum1_full[:, -mat.shape[1] :]

    # 取四个分支中均值最大的作为噪声水平
    noiselevel = np.max(trainMean, axis=0)

    coords = np.argwhere(mat / noiselevel > threshold)
    return coords, noiselevel


def cfar_1d(mat, numTrain, numGuard, threshold, type="mean"):

    mat = np.reshape(mat, (1, -1))
    shape = (1, 1 + 2 * numTrain + 2 * numGuard)
    convKernel = np.zeros(shape)

    # CA-CFAR,十字形状
    if type == "mean":
        convKernel[0, :numTrain] = 1
        convKernel[0, -numTrain:] = 1
        convKernel /= np.sum(convKernel)
    else:
        raise NotImplementedError("unKnown CFAR type.")
    noise_level = convolve2d(mat, convKernel, mode="same", boundary="wrap")
    coords = np.argwhere((mat / noise_level) > threshold)

    return coords[:, 1], noise_level.ravel()


def cfar_result_filtering(cfar_indices, range0, range1, shape1, th, mapping=[0, 1, 2]):
    if len(cfar_indices) == 0:
        return cfar_indices
    cfar = np.array(cfar_indices).astype(int)
    r0 = cfar[:, mapping[0]]
    mask_r0 = np.abs(r0[:, None] - r0) < range0
    np.fill_diagonal(mask_r0, False)

    r1 = cfar[:, mapping[1]].astype(int)
    mask_r1 = np.abs((r1[:, None] - r1) % shape1) < range1
    np.fill_diagonal(mask_r1, False)

    mag = cfar[:, mapping[2]]
    mask_mag = (mag[:, None] / mag) < th
    np.fill_diagonal(mask_mag, False)

    mask = np.sum(mask_r0 & mask_r1 & mask_mag, axis=1) == 0

    return [item for item, m in zip(cfar_indices, mask) if m]

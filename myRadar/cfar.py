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
    if type == "CrossMean":
        convKernel[: numTrain[0], np.floor_divide(shape[1], 2)] = 1
        convKernel[-numTrain[0] :, np.floor_divide(shape[1], 2)] = 1
        convKernel[np.floor_divide(shape[0], 2), : numTrain[1]] = 1
        convKernel[np.floor_divide(shape[0], 2), -numTrain[1] :] = 1
        convKernel /= np.sum(convKernel)
        noise_level = convolve2d(mat, convKernel, mode="same", boundary="wrap")
        coords = np.argwhere(mat / noise_level > threshold)
    elif type == "CrossMaxMean":
        coords, noise_level = cfar_2d_cross_prefix(mat, numTrain, numGuard, threshold)
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

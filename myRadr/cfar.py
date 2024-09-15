import numpy as np
from scipy.signal import convolve2d


def cfar_2d(mat, numTrain, numGuard, threshold, type="Cross"):
    numTrain = np.array(numTrain)
    numGuard = np.array(numGuard)

    shape = np.array([1, 1]) + 2 * numTrain + 2 * numGuard
    convKernel = np.zeros(shape)

    # CA-CFAR,十字形状
    if type == "Cross":
        convKernel[: numTrain[0], np.floor_divide(shape[1], 2)] = 1
        convKernel[-numTrain[0] :, np.floor_divide(shape[1], 2)] = 1
        convKernel[np.floor_divide(shape[0], 2), : numTrain[1]] = 1
        convKernel[np.floor_divide(shape[0], 2), -numTrain[1] :] = 1
        convKernel /= np.sum(convKernel)
    else:
        raise NotImplementedError("unKnown CFAR type.")
    noise_level = convolve2d(mat, convKernel, mode="same", boundary="wrap")
    coords = np.argwhere(mat / noise_level > threshold)

    return coords, noise_level

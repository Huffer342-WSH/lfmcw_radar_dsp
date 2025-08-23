import numpy as np
import scipy.io


class AT24G_RawData_RealI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-RawData-RealI16":
            return
        info = np.frombuffer(data[:6], dtype=np.uint8)
        numChannel = info[1]
        numChirp = int(info[2]) + int(info[3]) * 256
        numSample = int(info[4]) + int(info[5]) * 256
        temp = np.frombuffer(data[6:], dtype=np.int16).reshape(numChirp, numSample, numChannel).transpose(2, 0, 1)
        obj = temp.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "AT24G-RawData-RealI16":
            raise Exception("type error")
        self.data_type = "AT24G_RawData_RealInt16"
        info = np.frombuffer(data, dtype=np.uint8)
        self.idxFrame = info[0]
        self.numChirp = int(info[2]) + int(info[3]) * 256
        self.numSample = int(info[4]) + int(info[5]) * 256
        self.numChannel = info[1]


class AT24G_RangeFFT_ComplexI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-RangeFFT-ComplexI16":
            return
        info = np.frombuffer(data[:4], dtype=np.uint8)
        numChannel = info[1]
        numRangeBin = info[2]
        numChirp = info[3]
        temp = np.frombuffer(data[4:], dtype=np.int16)
        complexData = temp[0::2] + 1j * temp[1::2]
        # print(f"shape:{numChannel},{numRangeBin} {numChirp}")
        complexData = complexData.reshape(numChannel, numRangeBin, numChirp).transpose(0, 2, 1)
        obj = complexData.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "AT24G-RangeFFT-ComplexI16":
            raise Exception("type error")
        self.data_type = "AT24G-RangeFFT-ComplexI16"
        info = np.frombuffer(data[:4], dtype=np.uint8)
        self.idxFrame = info[0]
        self.numChannel = info[1]
        self.numRangeBin = info[2]
        self.numChirp = info[3]


class AT24G_2DFFT_ComplexI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-2DFFT-ComplexI16":
            raise ValueError("Invalid type provided, expected 'AT24G-2DFFT-ComplexI16'")

        # Extract metadata
        info = np.frombuffer(data[:4], dtype=np.uint8)
        idxFrame, numChannel, numRangeBin, numChirp = info

        # Parse complex data
        temp = np.frombuffer(data[4:], dtype=np.int16)
        complexData = temp[0::2] + 1j * temp[1::2]
        complexData = complexData.reshape(numChannel, numRangeBin, numChirp).transpose(0, 2, 1)

        # Create ndarray view
        obj = complexData.view(cls)
        obj.idxFrame = idxFrame
        obj.numChannel = numChannel
        obj.numRangeBin = numRangeBin
        obj.numChirp = numChirp
        obj.data_type = type

        return obj


def saveChirps(chirps, filename: str):
    savedata = dict()
    savedata["data_type"] = chirps[0].data_type
    i = 0
    while chirps[i].idxChirp != 0:
        chirps_num = chirps[i].idxChirp
        i = i + 1
    chirps_num = chirps_num + 1
    frame_num = (len(chirps) - i) // chirps_num
    radar_data = chirps[i : i + chirps_num * frame_num]
    savedata = {"radar_data": radar_data, "data_type": chirps[0].data_type}
    scipy.io.savemat(filename, savedata, do_compression=True)

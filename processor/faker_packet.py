# %%
import threading, io, typing, logging, time
import numpy as np
from collections import deque
import queue

import scipy.io

from . import base
from . import datapacket
from .usart import Usart

# %%


class McuPacket_Manager(threading.Thread, base.BaseLogger):
    """数据包结构
    +-------------------------------------+
    | INDEX   |  0   |  1   |  2   |  3   |
    |---------|------|------|------|------|
    | CONTENT | HEAD | INFO | DATA | TAIL |
    +-------------------------------------+
    HEAD: 固定的一段数据
    INFO: 数据包头部的信息字段
    DATA: 数据字段
    TAIL: 可包含校验和或者固定的尾
    """

    HEAD = bytearray([0x69, 0x6E, 0x66, 0x6F, 0x68, 0x61, 0x6E, 0x64, 0x6D, 0x6D, 0x52, 0x61, 0x64, 0x61, 0x72, 0x73])

    def __init__(self, port, baudrate, queue: queue.Queue) -> None:
        """
        初始化线程对象，继承自threading.Thread，可指定数据流及处理该流数据的回调函数。

        参数:
            - stream: 可选, 默认为None, 类型为io.IOBase。指定的数据流，用于线程内部处理。
            - callback: 可选, 默认为None, 类型为Callable[[bytes], bool]。当有数据处理需求时调用的函数，应接受字节串并返回布尔值。

        返回:
            无
        """
        threading.Thread.__init__(self, daemon=True)
        base.BaseLogger.__init__(self, level=logging.DEBUG)

        self.packet_queue = queue
        self.rdms = scipy.io.loadmat(file_name="E:/Projects/MCU/AT24GHz/tools/lfmcw_radar_dsp/data/AT24G_RecordedData_运动人体_长方形轨迹.mat")[
            "RDM"
        ].transpose(0, 1, 3, 2)
        self.numFrame, self.numChannel, self.numRangeBin, self.numChirp = self.rdms.shape
        self.numSample = 256

        rdms_padded = np.zeros(shape=(self.numFrame, self.numChannel, 256, self.numChirp),dtype=np.complex64)
        rdms_padded[:, :, : self.numRangeBin, :] = self.rdms
        self.raw = np.fft.ifft2(rdms_padded, axes=(-2, -1))

    def gen_rawdata_real_i16(self, idxFrame=0):
        """
        生成符合 AT24G-RawData-RealI16 类所需格式的字节数据
        """
        numChannel = self.numChannel
        numChirp = self.numChirp
        numSample = self.numSample

        # 头部信息 (6字节)
        header = bytearray(6)
        header[0] = idxFrame & 0xFF
        header[1] = numChannel & 0xFF
        header[2] = numChirp & 0xFF
        header[3] = (numChirp >> 8) & 0xFF
        header[4] = numSample & 0xFF
        header[5] = (numSample >> 8) & 0xFF

        # 分配数组
        raw = self.raw[idxFrame % self.numFrame].real
        raw = raw * 20000 / np.max(np.abs(raw))
        raw = raw.transpose(2,1,0).astype(np.int16)

        # 拼接 header + 数据
        data = bytes(header) + raw.tobytes()
        return data

    def gen_2dfft_complex_i16(self, idxFrame):
        _2dfft = self.rdms[idxFrame % self.numFrame]

        numChannel, numRangeBin, numChirp = _2dfft.shape

        iq_int16 = np.empty((numChannel, numRangeBin, numChirp, 2), dtype=np.int16)
        iq_int16[..., 0] = (_2dfft.real).astype(np.int16)
        iq_int16[..., 1] = (_2dfft.imag).astype(np.int16)

        # 头部
        header = np.array([idxFrame % 256, numChannel, numRangeBin, numChirp], dtype=np.uint8).tobytes()

        # 展平成 int16 流
        payload = iq_int16.tobytes()

        data = header + payload
        return data

    def run(self):
        idxFrame = 1

        while True:
            idxFrame += 1
            self.packet_queue.put(("AT24G-RawData-RealI16", self.gen_rawdata_real_i16(idxFrame)))
            self.packet_queue.put(("AT24G-2DFFT-ComplexI16", self.gen_2dfft_complex_i16(idxFrame)))
            time.sleep(0.1)

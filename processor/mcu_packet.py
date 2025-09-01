# %%
import threading, io, typing, logging, time
import numpy as np
from collections import deque
import queue

from scipy.io import savemat

from . import base
from . import datapacket
from .usart import Usart

# %%


def print_bytearray(byte_array):
    try:
        # 尝试将 bytearray 转换为 UTF-8 字符串
        string_data = byte_array.decode("utf-8")
        print(string_data)
    except UnicodeDecodeError:
        # 如果转换失败，打印十六进制表示
        hex_data = byte_array.hex()
        print(hex_data)


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

        self.port = port
        self.baudrate = baudrate

        self.packet_queue = queue
        self.que = deque(maxlen=1000)
        self._prev_packet_index = -1

    def stop(self) -> None:
        self._isRunning = False

    def save(self, file_path) -> None:
        savemat(file_path, {"deque_data": self.que}, do_compression=True)

    def registerCallback(self, callback: typing.Callable[[bytes], bool]) -> None:
        self.callback = callback

    def registerStream(self, stream: io.IOBase) -> None:
        self.stream = stream

    def checkMcuPacket(arr: bytes) -> tuple[bool, str]:
        reason = ""
        sizeTail = 4
        ret = True

        if len(arr) < 8:
            # 检查长度
            ret = False
            reason = "too short"

            print("[McuPacket_Manager]: (ERROR) too short")
        else:
            check_sum_inside = int.from_bytes(arr[-4:], byteorder="little")
            check_sum_outside = np.sum(np.frombuffer(arr[0:-sizeTail], dtype=np.uint8), dtype=np.uint32)
            if check_sum_inside != check_sum_outside:
                reason = f"check sum wrong : except {check_sum_inside} but get {check_sum_outside}"
                ret = False
            else:
                ret = True
        return ret, reason

    def run(self):

        self.stream = Usart(port=self.port, baudrate=self.baudrate)
        self.stream.start()
        self._isRunning = True

        sizeHead = len(self.HEAD)
        sizeInfo = 36
        sizeTail = 4

        ST_HEAD = 0  # 查找包头阶段
        ST_INFO = 1  # 查找数据包头部的信息字段
        ST_DATA = 2  # 接续数据字段并校验

        TYPE_MAP = {
            0: "AT24G-RawData-RealI16",
            1: "AT24G-RangeFFT-ComplexI16",
            2: "AT24G-2DFFT-ComplexI16",
            3: "TrackedObjectInfo",
            4: "RawData_RealI16",
        }

        state = ST_HEAD

        buffer = bytearray()
        buffer.extend(self.stream.read())
        skippedByte = bytearray()
        numSkippedBytes = 0
        while self._isRunning:
            if state == ST_HEAD:
                """寻找数据包头部。缓存等于数据包头长度的数据，判断是否为数据包的头部"""
                if len(buffer) < sizeHead:
                    buffer.extend(self.stream.read(sizeHead - len(buffer)))
                i = 0
                while state == ST_HEAD and i + sizeHead <= len(buffer):
                    if buffer[i : i + sizeHead] == McuPacket_Manager.HEAD:
                        if numSkippedBytes > 0:
                            try:
                                s = skippedByte.decode("utf-8")
                            except UnicodeDecodeError:
                                s = skippedByte.hex()
                            skippedByte = bytearray()
                            numSkippedBytes = 0
                        state = ST_INFO
                    else:
                        i += 1
                        skippedByte.extend(buffer[i - 1 : i])
                        numSkippedBytes += 1
                if i != 0:
                    del buffer[0:i]
            elif state == ST_INFO:
                """读取数据包的头部信息，检查参数是否合理"""
                if len(buffer) < sizeInfo + sizeHead:
                    buffer.extend(self.stream.read(sizeInfo + sizeHead - len(buffer)))
                info = buffer[sizeHead : sizeHead + sizeInfo]
                _type = info[27]
                _frameID = int.from_bytes(info[28:32], byteorder="little")
                _dataSize = int.from_bytes(info[32:36], byteorder="little")
                packetSize = sizeHead + sizeInfo + _dataSize + sizeTail

                self.log_debug(f"Get packet info, type: {TYPE_MAP[_type]}, frameID: {_frameID}, dataSize: {_dataSize}")
                state = ST_DATA
            elif state == ST_DATA:
                """解析数据包，合法则回调，否则跳过该数据包"""
                # 根据packet_size读取剩余数据
                if len(buffer) < packetSize:
                    buffer.extend(self.stream.read(packetSize - len(buffer)))

                # 校验数据包
                flag, reason = McuPacket_Manager.checkMcuPacket(buffer[0:packetSize])
                if flag == False:
                    self.log_warning(f'Packet Wrapper ERROR, reason: "{reason}"')
                else:
                    self.log_info(f"Packet Wrapper OK")

                # 回调成功
                if flag == True:
                    _data = buffer[sizeHead + sizeInfo : sizeHead + sizeInfo + _dataSize]
                    try:
                        self.packet_queue.put_nowait((TYPE_MAP[_type], _frameID, _data))
                    except queue.Full:
                        self.log_warning("packet queue is full")

                    del buffer[0:packetSize]  # 缓冲区删除一个包
                    if len(buffer) > 20000:
                        self.log_warning(f"Parsing speed cannot keep up with input speed")
                else:
                    i = 1
                    while i < len(buffer) and buffer[i] != self.HEAD:
                        i += 1
                    del buffer[0:i]
                    self.log_debug(f"The packet is parsed incorrectly, and {i} bytes are discarded")
                state = ST_HEAD
        self._isRunning = False


queuePacket = deque(maxlen=1000)


def parse(type: str, data: bytes):
    if type == "AT24G-RawData-RealI16":
        packet = datapacket.AT24G_RawData_RealI16(type, data)
        queuePacket.append(packet)
    elif type == "AT24G-RangeFFT-ComplexI16":
        packet = datapacket.AT24G_RangeFFT_ComplexI16(type, data)
        queuePacket.append(packet)
    elif type == "AT24G-2DFFT-ComplexI16":
        packet = datapacket.AT24G_2DFFT_ComplexI16(type, data)
        queuePacket.append(packet)
    else:
        print("unknown type: ", type)
        return False
    return packet


def task_recv_packet(queue: queue.Queue):
    while True:
        type, data = queue.get()
        packet = parse(type, data)
        if packet is not None:
            print(f"Receive packet: {packet.__dict__}")


if __name__ == "__main__":

    serial_port = "/dev/ttyUSB2"
    serial_port = Usart.select_serial_port()

    packet_queue = queue.Queue(maxsize=32)
    thread_parse = McuPacket_Manager(serial_port, 3250000, queue=packet_queue)
    thread_parse.logger.setLevel(logging.DEBUG)
    thread_parse.daemon = True

    thread_recv_packet = threading.Thread(target=task_recv_packet, args=(thread_parse.packet_queue,))

    thread_recv_packet.start()
    thread_parse.start()

    try:
        while True:
            user_input = input()  # 阻塞等待输入
            if user_input == "save":
                print("Saving data...")
                savemat("./RecordedData.mat", {"deque_data": queuePacket}, do_compression=True)
                print("Data saved.")
            elif user_input == "exit":
                print("Exiting input thread...")
                break  # 退出输入线程
    except KeyboardInterrupt:
        print("exit")
        thread_parse.stop()
        thread_parse.join()

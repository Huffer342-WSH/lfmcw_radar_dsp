import numpy as np
import scipy.io


class AT24G_RawData_RealI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-RawData-RealI16":
            return
        info = np.frombuffer(data[:6], dtype=np.uint8)
        numChannel = info[1]
        numChrip = info[2] + info[3] * 256
        numSample = info[4] + info[5] * 256
        temp = np.frombuffer(data[6:], dtype=np.int16).reshape(numChrip, numSample, numChannel).transpose(2, 0, 1)
        obj = temp.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "AT24G-RawData-RealI16":
            raise Exception("type error")
        self.data_type = "AT24G_RawData_RealInt16"
        info = np.frombuffer(data, dtype=np.uint8)
        self.idxFrame = info[0]
        self.numSample = info[4] + info[5] * 256
        self.numChrip = info[2] + info[3] * 256
        self.numChannel = info[1]


class AT24G_RangeFFT_ComplexI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-RangeFFT-ComplexI16":
            return
        info = np.frombuffer(data[:4], dtype=np.uint8)
        numChannel = info[1]
        numRangeBin = info[2]
        numChrip = info[3]
        temp = np.frombuffer(data[4:], dtype=np.int16)
        complexData = temp[0::2] + 1j * temp[1::2]
        # print(f"shape:{numChannel},{numRangeBin} {numChrip}")
        complexData = complexData.reshape(numChannel, numRangeBin, numChrip).transpose(0, 2, 1)
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
        self.numChrip = info[3]


class AT24G_2DFFT_ComplexI16(np.ndarray):
    def __new__(cls, type, data):
        if type != "AT24G-2DFFT-ComplexI16":
            return
        info = np.frombuffer(data[:4], dtype=np.uint8)
        numChannel = info[1]
        numRangeBin = info[2]
        numChrip = info[3]
        temp = np.frombuffer(data[4:], dtype=np.int16)
        complexData = temp[0::2] + 1j * temp[1::2]
        # print(f"shape:{numChannel},{numRangeBin} {numChrip}")
        complexData = complexData.reshape(numChannel, numRangeBin, numChrip).transpose(0, 2, 1)
        obj = complexData.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "AT24G-2DFFT-ComplexI16":
            raise Exception("type error")
        self.data_type = "AT24G-2DFFT-ComplexI16"
        info = np.frombuffer(data[:4], dtype=np.uint8)
        self.idxFrame = info[0]
        self.numChannel = info[1]
        self.numRangeBin = info[2]
        self.numChrip = info[3]


class RawData_RealI16(np.ndarray):
    def __new__(cls, type, data):

        if type != "RawData_RealI16":
            return
        temp = np.frombuffer(data[4:], dtype=np.int16)
        obj = temp.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "RawData_RealI16":
            raise Exception("type error")
        self.data_type = "RAW_RealInt16"
        self.numPoints = data[0] + data[1] * 256
        self.idxFrame = data[2]
        self.idxChrip = data[3]
        # print(f"idxChrip:{self.idxChrip} idxFrame:{self.idxFrame}")
        return


class _1DFFT(np.ndarray):

    def __new__(cls, type, data):

        if type != "1DFFT":
            return
        temp = np.frombuffer(data[4:], dtype=np.float32)
        temp = temp[::2] + 1.0j * temp[1::2]
        temp = temp.astype(np.complex128)
        obj = temp.view(cls)
        return obj

    def __init__(self, type, data):
        if type != "1DFFT":
            raise Exception("type error")
        self.data_type = "1DFFT_Complexfloat32"
        self.numRangeBin = data[0] + data[1] * 256
        self.idxChrip = data[3]
        self.idxFrame = data[2]
        # print(f"idxChrip:{self.idxChrip} idxFrame:{self.idxFrame}")
        return


class MTITargetsList:

    def __init__(self, type, data):
        if type != "MtiTargetsList":
            raise Exception(f"type error:{type},expected:MtiTargetsList")
        self.data_type = "MtiTargetsList"
        info = np.frombuffer(data[0:4], dtype=np.uint16)
        self.idxFrame = info[0]
        self.numTarget = info[1]
        start = 4
        end = start + self.numTarget
        self.targets_rangeCellIndex = np.frombuffer(data[start:end], dtype=np.uint8)
        start = (end + 3) // 4 * 4
        end = start + self.numTarget * 4
        self.targets_velocity = np.frombuffer(data[start:end], dtype=np.float32)

    def __str__(self):
        output = f"Frame Index:\t{self.idxFrame}\n"
        if self.numTarget == 0:
            print("No Targets")
        else:
            output += f"Target\tRange\tVelocity\n"
            for i in range(self.numTarget):
                range_cell_index = self.targets_rangeCellIndex[i]
                velocity = self.targets_velocity[i]
                output += f"{i + 1}\t{range_cell_index}\t{velocity:.3f}\n"
        return output


class TrackedObjectInfo:
    __STATE_MAP = {0: "无目标", 1: "慢速", 2: "静止", 3: "远离", 4: "靠近"}

    def __init__(self, type, data):
        if type != "TrackedObjectInfo":
            raise Exception(f"type error: {type}, expected: TrackedObjectInfo")
        self.data_type = "TrackedObjectInfo"

        # Ensure the data is in bytes and of the expected length
        if len(data) != 32:
            raise ValueError(f"Data must be a bytes object of length 32, got {len(data)}")

        # Convert the bytes data to a numpy array for easier manipulation
        buffer = np.frombuffer(data, dtype=np.uint8)

        # Parse the data according to the provided C structure
        self.idxFrame = int.from_bytes(buffer[0:2], byteorder="little", signed=False)
        self.state = TrackedObjectInfo.__STATE_MAP.get(buffer[0], "ERROR")
        self.rangeCellIndex = buffer[3]

        self.cntSlow = int.from_bytes(buffer[4:6], byteorder="little", signed=False)
        self.cntStatic = int.from_bytes(buffer[6:8], byteorder="little", signed=False)

        float_buffer = np.frombuffer(data[8:], dtype=np.float32)

        self.positionCenter = float_buffer[0]
        self.positionOffset = float_buffer[1]
        self.positionFiltered = float_buffer[2]
        self.velocity = float_buffer[3]
        self.complexData = float_buffer[4] + 1.0j * float_buffer[5]

    def __repr__(self):
        return (
            f"TrackedObjectInfo(frameIdx={self.idxFrame}, state={self.state}, "
            f"rangeCellIndex={self.rangeCellIndex}, cntSlow={self.cntSlow}, "
            f"cntStatic={self.cntStatic}, positionCenter={self.positionCenter}, "
            f"positionOffset={self.positionOffset}, positionFiltered={self.positionFiltered}, "
            f"velocity={self.velocity}, complexData={self.complexData} "
        )


class S3KM111L(np.ndarray):
    def __new__(cls, arr):
        # 获取数据部分，构建numpy数组
        temp = np.frombuffer(arr[4:-4], dtype=np.int16)
        temp = temp[::2] + 1.0j * temp[1::2]
        obj = temp.view(cls)
        return obj

    def __init__(self, arr):
        arr = np.frombuffer(arr, dtype=np.uint8)

        # 提取数据类型
        data_type = (arr[0] & 0b01110000) >> 4
        if data_type == 2:
            self.data_type = "dsraw"
        elif data_type == 3:
            self.data_type = "1dfft"
        elif data_type == 4:
            self.data_type = "2dfft"

        # 提取chrip编号
        self.idxChrip = (arr[3] >> 3) + ((arr[0] & 0b1111) << 5)

        # 提取数据长度
        len = (+arr[2] - 1) * 1
        if self.__len__() != len:
            print("数据包有误")
            print(len, self.__len__())

        return

    #  静态方法 校验数据包
    def checkPacket(pack: bytearray):
        pack = np.asarray(pack, dtype=np.uint8)
        # 确认头尾
        if pack[0] != 0xAA or pack[-1] != 0x55:
            print("头尾错误")
            return False
        # 长度校验
        data_len = ((pack[2] & 0b00000111) << 8) + pack[3] - 1
        if 4 * (data_len + 2) != len(pack):
            print("长度错误")
            return False
        # 确认校验和
        check_sum = pack[-4:-2].view(np.uint16)
        data = pack[4:-4].view(np.uint16)
        if check_sum[0] != np.sum(data, dtype=np.uint16):
            print("校验和错误")
            return False
        return True

    def parseMcuPacket_from_bytearray(data: bytearray):
        """
        解析MCU传来的数据包。

        参数:
        - data: 接收到的数据流(bytearray)。

        返回值:
        - packets: 解析出的数据包列表。
        - rest_data: 解析完成后剩余未处理的数据。
        """
        # 定义包头和包尾
        HEAD = bytearray([0x49, 0x43, 0x4C, 0x48])
        TAIL = bytearray([0x49, 0x43, 0x4C, 0x54])

        # 定义状态变量，用于指示当前解析阶段
        ST_HEAD = 0  # 查找包头阶段
        ST_PARA = 1  # 解析包参数阶段
        ST_FIRST_FIND = 2  # 找到第一个完整包阶段
        ST_SPLIT = 3  # 分割数据包阶段
        ST_WRONG = 4  # 包错误阶段
        ST_EMPTY = 5  # 数据为空阶段

        state: int = 0  # 初始化状态为查找包头阶段
        index = 0  # 初始化索引
        packets = []  # 初始化数据包列表
        rest_data = bytearray()  # 初始化剩余数据

        # 主循环，直到解析完成或数据为空
        while state != ST_EMPTY:
            if state == ST_HEAD:
                # 在数据中查找包头
                try:
                    index = data.index(HEAD)
                    state = ST_PARA
                except ValueError:
                    # 如果未找到包头，则重置状态为数据为空
                    print("未找到包头")
                    state = ST_EMPTY
            elif state == ST_PARA:
                # 解析包参数，计算包长度并验证数据完整性
                if data.__len__() - index < 12:
                    state = ST_EMPTY
                    break
                data_len = data[index + 4] + data[index + 5] * 256
                pack_len = data_len + 12
                pack_type = data[index + 6]
                channel = data[index + 7]
                if data.__len__() < index + 12 + data_len:
                    state = ST_EMPTY
                    break
                tail = data[index + 8 + data_len : index + 12 + data_len]
                if tail == TAIL:
                    state = ST_FIRST_FIND
                else:
                    state = ST_WRONG
            elif state == ST_FIRST_FIND:
                # 找到第一个完整包，准备进行数据包分割
                data = data[index:]
                state = ST_SPLIT
            elif state == ST_SPLIT:
                # 分割数据包，并对每个包进行处理
                index = 0
                while index + pack_len < data.__len__():
                    if data[index : index + 4] != HEAD or data[index + 8 + data_len : index + 12 + data_len] != TAIL:
                        data = data[index + 1 :]
                        state = ST_HEAD
                        break
                    channel = data[index + 7]
                    if not S3KM111L.checkPacket(data[index + 8 : index + 8 + data_len]):
                        print("数据包错误")
                    pack = S3KM111L(channel, data[index + 8 : index + 8 + data_len])
                    packets.append(pack)
                    index = index + pack_len
                if state == ST_HEAD:
                    continue
                rest_data = data[index:]
                return packets, rest_data
            elif state == ST_WRONG:
                # 如果包尾不正确，则丢弃当前包，重新查找包头
                data = data[4:]
                state = ST_HEAD


def saveChrips(chrips, filename: str):
    savedata = dict()
    savedata["data_type"] = chrips[0].data_type
    i = 0
    while chrips[i].idxChrip != 0:
        chrips_num = chrips[i].idxChrip
        i = i + 1
    chrips_num = chrips_num + 1
    frame_num = (len(chrips) - i) // chrips_num
    radar_data = chrips[i : i + chrips_num * frame_num]
    savedata = {"radar_data": radar_data, "data_type": chrips[0].data_type}
    scipy.io.savemat(filename, savedata, do_compression=True)

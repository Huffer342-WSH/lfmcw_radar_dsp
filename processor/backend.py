import multiprocessing
import multiprocessing.connection
import multiprocessing.synchronize
import threading
import time
import copy
import queue
import datetime
import logging
from collections import deque
from itertools import chain


import numpy as np
import scipy.io
import scipy.constants
from scipy.fft import fftshift

import plotly.graph_objects as go

from . import base
from .mcu_packet import McuPacket_Manager
from . import datapacket
from .usart import Usart
from .core import Processor, RadarInitParam, RadarConfig, RadarCFARConfig, RadarCFARFilterConfig, TrackConfig, DBSCANConfig, TrackedTarget


class BackEnd(multiprocessing.Process, base.BaseLogger):

    _type_map = {
        "AT24G-RawData-RealI16": "SignalRaw",
        "AT24G-RangeFFT-ComplexI16": "SignalRangeFFT",
        "AT24G-2DFFT-ComplexI16": "Signal2DFFT",
    }

    _callback_map = {
        "AT24G-RawData-RealI16": datapacket.AT24G_RawData_RealI16,
        "AT24G-RangeFFT-ComplexI16": datapacket.AT24G_RangeFFT_ComplexI16,
        "AT24G-2DFFT-ComplexI16": datapacket.AT24G_2DFFT_ComplexI16,
    }

    def __init__(
        self,
        message_queue: multiprocessing.Queue,
        conn: multiprocessing.connection._ConnectionBase,
        event_shutdown: multiprocessing.synchronize.Event,
        serial_config,
    ):
        multiprocessing.Process.__init__(self)
        base.BaseLogger.__init__(self)

        self.message_queue = message_queue
        self.conn = conn
        self.event_shutdown = event_shutdown
        self.serial_config = serial_config

        self.is_init = False
        self.cntFrame: int = 0
        self.idxFrame = 0

        self.numSample = 0
        self.numRangeBin = 15
        self.numChrip = 0

        self.__tempFrame = {}

        self._hasRaw = False
        self._hasRangeFFT = False
        self._has2DFFT = False

    def __initialize(self):
        self.packet_queue = queue.Queue(maxsize=32)
        self.frame_queue = queue.Queue(maxsize=32)
        self.bufferFrame = deque(maxlen=2000)

        self.processor = self.creat_radar_processor()

    def creat_radar_processor(self) -> Processor:
        processor = Processor(
            param=RadarInitParam(
                wavelength=scipy.constants.c / 24.125e9,
                bandwidth=204e6,
                rx_antenna_spacing=6.98e-3,
                timeChrip=2.461538e-5,
                timeChripGap=1.310367e-03,
                timeFrameGap=4.561123e-03,
                numChannel=2,
                numRangeBin=16,
                numChrip=64,
                numMaxCfarPoints=64,
                numMaxCachedFrame=8,
                numInitialMultiMeas=4,
                numInitialCluster=4,
            ),
            config=RadarConfig(
                cfar_cfg=RadarCFARConfig(numTrain=(3, 8), numGuard=(2, 4), thSNR=2.5, thMag=500),
                cfar_filter_cfg=RadarCFARFilterConfig(range0=2, range1=3, shape1=64, th=0.8),
                dbscan_cfg=DBSCANConfig(wr=1, wv=2, eps=0.6, min_samples=5),
                track_cfg=TrackConfig(
                    tran_model_q=5,
                    meas_noise_r=np.diag([0, 5 / 180 * np.pi, 0.2, 0.07]) ** 2,
                    missed_distance=2.5,
                    del_unassociated_time=5.0,
                    del_missed_probability=0.75,
                    init_unassociated_time=2.0,
                    init_keep_motion_time=1.0,
                    init_keep_static_time=10.0,
                    init_speed_th=0.03,
                    init_missed_distance=2.0,
                    init_covar=np.diag([0.7, 0.2, 0.7, 0.2, 0.7, 0.2]) ** 2,
                    fov=np.array([-np.pi * 45 / 180, np.pi * 45 / 180]),
                    radius_range=np.array([0.3, 8.0]),
                ),
                channel_phase_diff_threshold=np.pi * 0.9,
            ),
        )
        return processor

    def update_idxFrame(self, idxFrame):
        idxFrame &= 0xFF
        if self.idxFrame & 0xFF != idxFrame:
            self.cntFrame += 1
            high = self.idxFrame >> 8
            if idxFrame == 0:
                high += 1
            self.idxFrame = (high << 8) + idxFrame
            return True
        else:
            return False

    def receiveOnePacket(self, type: str, data: bytes):
        packet = None
        func = self._callback_map.get(type, None)
        if func is not None:
            packet = func(type, data)
        else:
            self.log_warning(f"Unsupported type: {type}")
            return False
        ret = None
        if self.update_idxFrame(packet.idxFrame):
            self.__tempFrame["idxFrame"] = self.idxFrame
            ret = copy.deepcopy(self.__tempFrame)

        self.__tempFrame[self._type_map[type]] = packet
        return ret

    def genFigure_2DFFT(self, rdm: np.ndarray):

        self.processor(rdm, timestamp=datetime.datetime.now())

        for target in self.processor.tracked_targets:
            target: TrackedTarget
            self.log_debug(f"Target {target.uuid} socre: {target.life_cycle.score}")

        msg = dict()

        # 图1 幅度谱
        magSepc2D = fftshift(np.sum(np.abs(rdm), axis=0).T, axes=0)

        msg["fig0"] = {
            "fig": go.Figure(
                data=go.Heatmap(z=magSepc2D),
                layout=go.Layout(title="RDM"),
            )
        }

        # 图2 点云
        figure_data = []

        # 2.1 点云
        measurements = np.array(list(chain.from_iterable(self.processor.basic.multi_frame_meas)))
        if measurements.ndim == 2:
            x = measurements[:, 1] * np.cos(measurements[:, 0])
            y = measurements[:, 1] * np.sin(measurements[:, 0])
            figure_data.append(
                go.Scatter(
                    x=y,
                    y=x,
                    mode="markers",
                    marker=dict(
                        symbol="star-triangle-up",
                        size=4,
                        color="rgba(62,143,230,1)",
                    ),
                    name="测量值",
                )
            )

        # 2.2 聚类
        points = np.array([[m[0], m[1]] for m in self.processor.basic.measurements])
        if points.ndim == 2:
            x = points[:, 1] * np.cos(points[:, 0])
            y = points[:, 1] * np.sin(points[:, 0])
            figure_data.append(
                go.Scatter(
                    x=y,
                    y=x,
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=10,
                    ),
                    line=dict(width=3, color="rgba(79,89,238,0.8)"),
                    name="测量值",
                )
            )

        # 2.2 起始阶段目标
        targets = self.processor.unconfirmed_targets
        x = np.array([target.state.state_vector[0, 0] for target in targets])
        y = np.array([target.state.state_vector[2, 0] for target in targets])
        figure_data.append(
            go.Scatter(
                x=y,
                y=x,
                mode="markers",
                name="已跟踪目标",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color="rgba(255,156,37,1)",
                ),
            )
        )
        # 2.2 已跟踪目标
        targets = self.processor.tracked_targets
        x = np.array([target.state.state_vector[0, 0] for target in targets])
        y = np.array([target.state.state_vector[2, 0] for target in targets])
        figure_data.append(
            go.Scatter(
                x=y,
                y=x,
                mode="markers",
                name="已跟踪目标",
                marker=dict(
                    symbol="circle",
                    size=8,
                    color="rgba(255,0,0,0.8)",
                ),
            )
        )

        msg["fig1"] = {
            "fig": go.Figure(
                data=figure_data,
                layout=go.Layout(
                    title="点云",
                    xaxis=dict(title="左右", range=[-4, 4], scaleanchor="y", scaleratio=1, constrain="domain"),
                    yaxis=dict(title="前后", range=[0, 10], scaleanchor="x", scaleratio=1, constrain="domain"),
                    legend=dict(orientation="h", entrywidth=70, yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                ),
            )
        }
        return msg

    def saveData(self) -> None:
        self.log_info("Saving data...")
        filename = datetime.datetime.now().strftime("AT24G_RecordedData %Y-%m-%d %H-%M-%S.mat")

        temp = list(self.bufferFrame)

        savedata = dict()
        savedata["RDM"] = np.stack([i["Signal2DFFT"] for i in temp])
        savedata.update(self.processor.param.__dict__)
        savedata["numFrame"] = len(temp)

        for i in temp:
            if type(i) != dict():
                self.log_warning(f"frame type error {type(i)}")

        savedata["frames"] = temp

        # 保存文件
        try:
            self.log_info(f"保存文件：{filename}")
            scipy.io.savemat(filename, savedata, do_compression=True)
        except Exception as e:
            self.log_error(f"保存文件失败：{e}")
            return

    def save2frame(self, frame: dict):
        frame["tracked_targets"] = [target.get_dict() for target in self.processor.tracked_targets]
        frame["unconfirmed_targets"] = [target.get_dict() for target in self.processor.unconfirmed_targets]

    def createThreadReceivePacket(self):
        def task():
            self.log_debug("接收数据线程启动")
            while self.event_shutdown.is_set() == False:
                packet_type, packet_data = self.packet_queue.get()
                self.log_debug(f"接收到数据包:{packet_type}")

                frame = self.receiveOnePacket(packet_type, packet_data)
                if frame is not None and self.is_init:

                    msg = dict()

                    rdm = frame.get("Signal2DFFT")
                    if rdm is not None:
                        temp = self.genFigure_2DFFT(rdm.transpose(0, 2, 1))
                        msg.update(temp)

                    # 发送数据
                    if not self.message_queue.full():
                        self.message_queue.put(msg)

                    self.save2frame(frame)

                    self.bufferFrame.append(frame)

        return threading.Thread(target=task, daemon=True)

    def run(self) -> None:
        self.__initialize()

        self.log_debug(f"进程启动 PID:{multiprocessing.current_process().pid}")

        self.mcuPackerManager = McuPacket_Manager(port=self.serial_config["name"], baudrate=self.serial_config["baudrate"], queue=self.packet_queue)
        self.mcuPackerManager.logger.setLevel(logging.WARNING)

        # 接收数据线程启动
        thread_recv_packet = self.createThreadReceivePacket()
        thread_recv_packet.start()
        self.mcuPackerManager.start()

        # 预接收一些数据
        while self.cntFrame < 3:
            time.sleep(0.1)
        self.is_init = True
        self.log_info(
            f"""\r\n
==========================================
预接收数据结束，雷达参数如下：
------------------------------------------
'Packets:'
{'\r\n'.join([f"{key}\t: {value.__dict__}" if hasattr(value, '__dict__') else f"{key}\t: {value}" for key, value in self.__tempFrame.items()])}
------------------------------------------
"""
        )

        self.log_info(f"雷达参数\r\n{self.processor.param}")
        self.log_info(f"雷达配置\r\n{self.processor.config}")

        # 开始实时处理数据
        while not self.event_shutdown.is_set():
            # 接收进程管道消息
            msg = self.conn.recv()
            self.log_info(f"Pipe receive message: {msg}")
            if msg["type"] == "save":
                self.saveData()
            else:
                self.log_warning(f"Unknown message: {msg}")

        self.log_info(f"等待线程结束 {thread_recv_packet.is_alive()}")
        thread_recv_packet.join()


if __name__ == "__main__":
    print("后端测试")
    from usart import Usart

    serial_port_name = Usart.select_serial_port()
    print(f"select serial port: {serial_port_name}")

    message_queue = multiprocessing.Queue(maxsize=3)
    event_shutdown = multiprocessing.Event()
    para, son = multiprocessing.Pipe()
    backend = BackEnd(message_queue=message_queue, conn=son, event_shutdown=event_shutdown, serial_config={"name": serial_port_name, "baudrate": 3250000})
    backend.start()

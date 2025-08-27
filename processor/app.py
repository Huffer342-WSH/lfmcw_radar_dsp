import multiprocessing, logging

from .frontend import FrontEnd
from .backend import BackEnd
from .usart import Usart


class Application:
    def __init__(self):
        # 初始化日志设置
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        # 初始化通信资源
        self.message_queue = multiprocessing.Queue(maxsize=3)
        self.event_shutdown = multiprocessing.Event()
        self.para, self.son = multiprocessing.Pipe(duplex=True)

        # 初始化串口配置
        # self.serial_config = {"name": "None", "baudrate": 0}  # 仿真数据，而非串口
        # self.serial_config = {"name": "COM17", "baudrate": 3250000} # 假如知道串口名可以写死
        self.serial_config = {
            "name": Usart.select_serial_port(),
            "baudrate": 3250000,
        }

        # 创建前端和后端对象
        self.app_wrapper = FrontEnd(queue=self.message_queue, conn=self.para)
        self.backend = BackEnd(
            message_queue=self.message_queue,
            conn=self.son,
            event_shutdown=self.event_shutdown,
            serial_config=self.serial_config,
            log_level=logging.WARNING,
        )

    def start(self, debug=False):
        """启动应用程序"""
        try:
            self.backend.start()  # 启动后端进程
            self.app_wrapper.run(debug=debug)  # 运行前端应用
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭应用程序"""
        self.event_shutdown.set()  # 通知后端进程关闭
        self.backend.join()  # 等待后端进程结束


# 如果此文件被直接运行
if __name__ == "__main__":
    app = Application()
    app.start(debug=False)

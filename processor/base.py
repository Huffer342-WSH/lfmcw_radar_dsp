import logging
import os
import inspect


class BaseLogger:

    def __init__(self, level=logging.DEBUG):
        # 获取当前类名
        class_name = self.__class__.__name__

        # 获取调用此类的文件名
        caller_frame = inspect.stack()[1]
        file_name = os.path.basename(caller_frame.filename)

        # 创建一个专属日志器
        self.logger = logging.getLogger(f"{file_name}::{class_name}")
        if not self.logger.hasHandlers():
            # 设置日志格式和处理器
            log_format = f"%(asctime)s - [%(levelname)s] - {file_name}::{class_name} - %(message)s"
            handler = logging.StreamHandler()  # 输出到控制台
            handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(handler)
            self.logger.setLevel(level)  # 默认等级为 DEBUG

    def log_debug(self, message):
        self.logger.debug(message)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_critical(self, message):
        self.logger.critical(message)

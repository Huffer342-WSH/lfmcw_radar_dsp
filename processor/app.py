import multiprocessing
import logging

from frontend import FrontEnd
from backend import BackEnd
from usart import Usart

if __name__ == "__main__":

    # 关闭Dash的日志
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    message_queue = multiprocessing.Queue(maxsize=3)
    event_shutdown = multiprocessing.Event()
    para, son = multiprocessing.Pipe(duplex=True)
    serial_config = {"name": "/dev/ttyUSB2", "baudrate": 3250000}
    # serial_config["name"] = Usart.select_serial_port()

    app_wrapper = FrontEnd(queue=message_queue, conn=para)
    backend = BackEnd(message_queue=message_queue, conn=son, event_shutdown=event_shutdown, serial_config=serial_config)
    backend.logger.setLevel(logging.INFO)
    backend.start()

    app_wrapper.run(debug=False)
    event_shutdown.set()
    backend.join()

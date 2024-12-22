import multiprocessing
from frontend import FrontEnd
from backend import BackEnd
import logging


if __name__ == "__main__":

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    message_queue = multiprocessing.Queue(maxsize=3)
    event_shutdown = multiprocessing.Event()
    para, son = multiprocessing.Pipe()

    app_wrapper = FrontEnd(
        queue=message_queue,
    )

    backend = BackEnd(message_queue=message_queue, conn=son, event_shutdown=event_shutdown, serial_config={"name": "COM17", "baudrate": 3250000})
    backend.start()

    app_wrapper.run(debug=False)
    event_shutdown.set()
    backend.join()

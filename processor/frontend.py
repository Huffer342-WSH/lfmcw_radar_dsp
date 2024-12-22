import plotly.graph_objects as go

import multiprocessing, threading, time
from datetime import datetime

import numpy as np

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State


import base


class FrontEnd(base.BaseLogger):
    def __init__(self, queue: multiprocessing.Queue, title="雷达上位机", update_title=None):
        """
        Initializes the Dash application.
        """
        self.app = Dash(__name__, title=title, update_title=update_title)
        self.parent_conn = None  # Placeholder for external connection
        self._setup_layout()
        self._setup_callbacks()
        self.threads = []
        # self.threads.append(self.create_thread())

        self.message_queue = queue
        self.fig_buffer = {"fig0": {"fig": go.Figure()}, "fig1": {"fig": go.Figure()}}

    def _setup_layout(self):
        """
        Sets up the layout of the Dash application.
        """
        self.app.layout = html.Div(
            children=[
                # 实时更新的间隔设置
                dcc.Interval(id="interval-component", interval=1000 / 10, n_intervals=120),
                dcc.Graph(id="fig0"),
                dcc.Graph(id="fig1", style={"height": "800px"}),
                # 按钮
                html.Button("⏸暂停", id="btn-pause", n_clicks=0, style={"margin-right": "10px"}),
                html.Button("保存数据", id="btn-savedata", n_clicks=0),
                html.Div(id="output-state"),
            ],
        )

    def _setup_callbacks(self):
        """
        Configures the callbacks for the Dash application.
        """

        @self.app.callback(
            [Output("fig0", "figure"), Output("fig1", "figure")],
            Input("interval-component", "n_intervals"),
        )
        def update_dashboard(n):
            if not self.message_queue.empty():
                self.fig_buffer = self.message_queue.get_nowait()

            # 图1  距离-时间

            fig0 = self.fig_buffer.get("fig0", {}).get("fig", go.Figure())
            fig1 = self.fig_buffer.get("fig1", {}).get("fig", go.Figure())

            return (fig0, fig1)

        @self.app.callback(
            [
                Output("btn-pause", "children"),
                Output("interval-component", "disabled"),
            ],
            Input("btn-pause", "n_clicks"),
        )
        def update_interval(n_clicks):
            if n_clicks % 2 == 0:
                return "⏯停止", False
            else:
                return "⏯继续", True

        @self.app.callback(
            Output("output-state", "children"),
            Input("btn-savedata", "n_clicks"),
        )
        def button_click(n_clicks):
            if n_clicks >= 1:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if self.parent_conn:
                    self.parent_conn.send({"type": "save"})
                return f"[{current_time}]: 已保存"
            return ""

    def create_thread(self):
        def background_task():
            while True:
                data = self.message_queue.get(timeout=10)
                self.log_debug(f"receive data: {data}")
                print(f"DashAppWrapper receive data")
                self.fig_buffer = data

        return threading.Thread(target=background_task, daemon=True)

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """
        Runs the Dash application.
        """

        base.BaseLogger.__init__(self)

        self.app.run(host=host, port=port, debug=debug)

        print("DashAppWrapper exit")


class BackEnd_Example(multiprocessing.Process):
    def __init__(self, message_queue: multiprocessing.Queue):
        multiprocessing.Process.__init__(self)
        self.message_queue = message_queue

    def run(self):
        print("BackEnd_Example start")
        while True:
            time.sleep(2)
            data = {
                "fig0": {"fig": go.Figure(data=go.Heatmap(z=np.random.rand(10, 10)))},
                "fig1": {"fig": go.Figure(data=go.Scatter(y=np.random.rand(10)))},
            }
            if self.message_queue.full():
                self.message_queue.get()
            self.message_queue.put(data)


# Example usage
if __name__ == "__main__":
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print("前端测试")

    message_queue = multiprocessing.Queue(maxsize=3)
    event_shutdown = multiprocessing.Event()
    para, son = multiprocessing.Pipe()

    app_wrapper = FrontEnd(
        queue=message_queue,
    )

    backend = BackEnd_Example(message_queue=message_queue)

    app_wrapper.run(debug=False)
    print("Dash App 运行结束")
    event_shutdown.set()

    backend.join()

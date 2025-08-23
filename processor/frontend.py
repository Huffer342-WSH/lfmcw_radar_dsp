import plotly.graph_objects as go

import multiprocessing, threading, time
import multiprocessing.connection
from datetime import datetime

import numpy as np

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

from . import base


class FrontEnd(base.BaseLogger):

    def __init__(self, queue: multiprocessing.Queue, conn: multiprocessing.connection._ConnectionBase, title="雷达上位机", update_title=None):
        """
        Initializes the Dash application.
        """
        self.app = Dash(__name__, title=title, update_title=update_title)
        self.parent_conn = conn
        self._setup_layout()
        self._setup_callbacks()

        self.message_queue = queue
        self.fig_buffer = {"fig0": {"fig": go.Figure()}, "fig1": {"fig": go.Figure()}}

    def _setup_layout(self):
        """
        Sets up the layout of the Dash application.
        """
        self.app.layout = html.Div(
            children=[
                dcc.Interval(id="interval-component", interval=1000 / 10, n_intervals=120),
                # 顶部下拉框控制区
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dropdown-fig0",
                            options=[
                                {"label": "原始数据(raw)", "value": "raw"},
                                {"label": "距离-多普勒(rdm)", "value": "rdm"},
                                {"label": "目标检测(target)", "value": "target"},
                                {"label": "不显示", "value": "none"},
                            ],
                            value="raw",
                            clearable=False,
                            style={"width": "30%", "display": "inline-block", "margin-right": "10px"},
                        ),
                        dcc.Dropdown(
                            id="dropdown-fig1",
                            options=[
                                {"label": "原始数据(raw)", "value": "raw"},
                                {"label": "距离-多普勒(rdm)", "value": "rdm"},
                                {"label": "目标检测(target)", "value": "target"},
                                {"label": "不显示", "value": "none"},
                            ],
                            value="none",
                            clearable=False,
                            style={"width": "30%", "display": "inline-block", "margin-right": "10px"},
                        ),
                        dcc.Dropdown(
                            id="dropdown-fig2",
                            options=[
                                {"label": "原始数据(raw)", "value": "raw"},
                                {"label": "距离-多普勒(rdm)", "value": "rdm"},
                                {"label": "目标检测(target)", "value": "target"},
                                {"label": "不显示", "value": "none"},
                            ],
                            value="none",
                            clearable=False,
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                # 图像容器
                html.Div(id="graph-container"),
                # 按钮区
                html.Div(
                    [
                        html.Button("⏸暂停", id="btn-pause", n_clicks=0, style={"margin-right": "10px"}),
                        html.Button("保存数据", id="btn-savedata", n_clicks=0),
                        html.Div(id="output-state"),
                    ],
                    style={"margin-top": "20px"},
                ),
            ],
        )

    def _setup_callbacks(self):
        """
        Configures the callbacks for the Dash application.
        """
        def default_figure(msg="无数据可展示"):
            fig = go.Figure()
            fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="red"))
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig

        @self.app.callback(
            Output("graph-container", "children"),
            [
                Input("interval-component", "n_intervals"),
                Input("dropdown-fig0", "value"),
                Input("dropdown-fig1", "value"),
                Input("dropdown-fig2", "value"),
            ],
        )
        def update_graphs(n, sel0, sel1, sel2):
            # 更新 buffer
            while self.message_queue.qsize() > 8:
                self.message_queue.get_nowait()
            if not self.message_queue.empty():
                self.fig_buffer = self.message_queue.get()

            def genGraph(selection: str):
                if selection == "none":
                    return default_figure()
                key_map = {"raw": ("fig_raw", 350), "rdm": ("fig_rdm", 350), "target": ("fig_target", 800)}

                fig_name, height = key_map.get(selection, (None, None))
                if fig_name is None:
                    return default_figure()

                packet = self.fig_buffer.get(fig_name)
                if packet is None:
                    return default_figure()

                fig = packet.get("fig")
                return dcc.Graph(id=f"fig{i}", figure=fig, style={"height": f"{height}px"})

            graphs = []
            for i, sel in enumerate([sel0, sel1, sel2]):
                if sel != "none":
                    graphs.append(genGraph(sel))
            return graphs

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
                self.parent_conn.send({"type": "save"})
                self.log_debug(f"Pipe sned message")
                return f"[{current_time}]: 已保存"
            return ""

    def run(self, host="127.0.0.1", port: str = "8050", debug=True):
        """
        Runs the Dash application.
        """

        base.BaseLogger.__init__(self)

        self.app.run(host=host, port=port, debug=debug)

        print("DashAppWrapper exit")


class BackEnd_Example(multiprocessing.Process):

    def __init__(self, message_queue: multiprocessing.Queue, conn: multiprocessing.connection._ConnectionBase):
        multiprocessing.Process.__init__(self)
        self.message_queue = message_queue
        self.conn = conn

    def task_recv_pipe(self):
        while True:
            data = self.conn.recv()
            print(f"BackEnd_Example receive data: {data}")

    def run(self):
        print("BackEnd_Example start")
        thread_recv_pipe = threading.Thread(target=self.task_recv_pipe, daemon=True)
        thread_recv_pipe.start()
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

    app_wrapper = FrontEnd(queue=message_queue, conn=para)

    backend = BackEnd_Example(message_queue=message_queue, conn=son)
    backend.start()

    app_wrapper.run(debug=False)
    print("Dash App 运行结束")
    event_shutdown.set()

    backend.join()

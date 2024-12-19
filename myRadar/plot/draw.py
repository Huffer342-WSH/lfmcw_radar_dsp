import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from moviepy.editor import ImageSequenceClip
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def draw_spectrum(spec, x=None, y=None, title: str = "default"):
    if x is None:
        x = np.arange(spec.shape[-1])
    if y is None:
        y = np.arange(spec.shape[-2])
    x_grid, y_grid = np.meshgrid(x, y)
    fig = go.Figure(data=[go.Surface(z=spec, x=x_grid, y=y_grid)])
    fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    fig.show()


def draw_2d_spectrumlist(z, x=None, y=None, title: str = "未命名") -> go.Figure:

    def create_frame(k):
        return go.Frame(
            data=go.Surface(x=x, y=y, z=z[k]),
            name=str(k),
        )

    nb_frames = len(z)
    fig = go.Figure(
        data=go.Surface(x=x, y=y, z=z[0]),
        frames=Parallel(n_jobs=-1)(delayed(create_frame)(k) for k in range(nb_frames)),
        # frames=[create_frame(k) for k in range(nb_frames)],
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "currentvalue": {"font": {"size": 20}, "prefix": "freme:", "visible": True, "xanchor": "right"},
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            zaxis=dict(autorange=True),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(projection=dict(type="orthographic")),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(10)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    return fig


def draw_scatter_list(y, x=None, title: str = "未命名", mode="markers"):
    def cals_lim(a):
        a_max = np.max(a)
        a_min = np.min(a)
        diff = a_max - a_min
        return (a_min - diff * 0.1, a_max + diff * 0.1)

    nb_frames = len(y)

    ylim = cals_lim(y)
    if x is None:
        x = [np.arange(len(k)) for k in y]
    xlim = cals_lim(x)

    fig = go.Figure(
        data=go.Scatter(x=x[0], y=y[0], mode=mode),
        frames=[
            go.Frame(
                data=go.Scatter(x=x[k], y=y[k], mode=mode),
                name=str(k),  # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ],
    )
    fig.update_layout(yaxis_range=ylim, xaxis_range=xlim)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "currentvalue": {"font": {"size": 20}, "prefix": "freme:", "visible": True, "xanchor": "right"},
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=title,
        # scene=dict(
        #     zaxis=dict(range=[0, 2e6], autorange=False),
        #     aspectratio=dict(x=1, y=1, z=1),
        # ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(10)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    return fig


def draw_animation(listData, title: str = "未命名") -> go.Figure:

    fig = go.Figure(
        data=listData[0],
        frames=[go.Frame(data=listData[k], name=str(k)) for k in range(len(listData))],
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "currentvalue": {"font": {"size": 20}, "prefix": "freme:", "visible": True, "xanchor": "right"},
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            # zaxis=dict(autorange=True),
            # aspectratio=dict(x=1, y=1, z=1),
            camera=dict(projection=dict(type="orthographic")),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(0)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    return fig


def draw_complex_Scatter3d(complex, title: str = "未命名"):
    fig = go.Figure(
        data=[go.Scatter3d(x=np.arange(len(complex)), y=np.real(complex), z=np.imag(complex), mode="lines+markers", marker=dict(size=5), line=dict(width=2))]
    )
    # 设置坐标轴比例相同
    fig.update_layout(
        scene=dict(
            aspectmode="manual", aspectratio=dict(x=5, y=1, z=1), xaxis=dict(title="Index"), yaxis=dict(title="Real Part"), zaxis=dict(title="Imaginary Part")
        ),  # 设置所有轴的比例相同
        title=title,
    )
    return fig

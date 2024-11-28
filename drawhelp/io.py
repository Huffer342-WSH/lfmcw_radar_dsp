# %%
import plotly.graph_objects as go
import ffmpeg
import os
import psutil
from multiprocessing import Queue, Process
from threading import Thread, Event
import tqdm
import heapq


__all__ = ["plotly_fig_to_video", "plotly_fig_to_video_multiprocess"]


def get_unique_filename(filepath):
    """
    检查文件是否存在，如果存在，给文件名加上后缀(1), (2)...直到找到一个不存在的文件名。
    """
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base}({counter}){ext}"
        counter += 1
    return new_filepath


def __generate_frame(fig: go.Figure, indices, queue: Queue):
    """
    生成单个帧的图像并将其通过队列传递给主进程。
    """
    for i in indices:
        fig.update(data=fig.frames[i].data)
        img_bytes = fig.to_image(format="png")
        queue.put((i, img_bytes))
    return


def plotly_fig_to_video_multiprocess(fig: go.Figure, output_path: str, fps: int = 30, width: int = None, height: int = None, njobs: int = -1):

    def store_farme_to_heap(heap, queue, total_frames):
        count = 0
        while count < total_frames:
            frame_index, img_bytes = queue.get()
            heapq.heappush(heap, (frame_index, img_bytes))
            event.set()
            count += 1

    output_path = get_unique_filename(output_path)
    layout = fig.layout
    fig.update_layout(dict1=dict(updatemenus=[], sliders=[]), overwrite=True)

    if njobs <= 0:
        njobs = psutil.cpu_count(logical=False)

    print(f"Number of Processes:{njobs}")

    fig.layout.width = width or fig.layout.width or 1080
    fig.layout.height = height or fig.layout.height or 607
    print(f"Image size: {fig.layout.width}x{fig.layout.height}")

    # 获取所有frames
    frames = fig.frames
    num_frames = len(frames)

    heap = []
    queue = Queue()  # 创建一个队列用于传递数据
    event = Event()

    # 启动多个进程生成图像
    processList = []
    for i in range(njobs):
        p = Process(target=__generate_frame, args=(fig, range(i, num_frames, njobs), queue))
        processList.append(p)
        p.start()

    # 启动一个线程将图像数据存储到堆
    theread_save_to_heap = Thread(target=store_farme_to_heap, args=(heap, queue, num_frames))
    theread_save_to_heap.start()

    # 使用ffmpeg进行视频编码，从pipe中读取图像
    process = (
        ffmpeg.input("pipe:0", framerate=fps, format="image2pipe", pix_fmt="yuv420p")  # 通过stdin传输图像
        .output(
            output_path,
            vcodec="h264_nvenc",
            cq=19,
        )
        .global_args("-loglevel", "warning")  # 设置日志级别为 quiet
        .run_async(pipe_stdin=True)  # 异步运行，打开stdin管道
    )

    # 按照编号顺序将图像数据传输给ffmpeg
    for i in tqdm.tqdm(range(num_frames)):
        while not heap or heap[0][0] != i:
            event.wait()
        event.clear()
        _, img_data = heapq.heappop(heap)
        process.stdin.write(img_data)
        del img_data

    # 关闭stdin，告诉ffmpeg输入已经完成
    process.stdin.close()

    for p in processList:
        p.join()
    theread_save_to_heap.join()
    process.wait()
    fig.layout = layout
    print(f"Video saved to {output_path}")


def plotly_fig_to_video(fig: go.Figure, output_path: str, fps: int = 30, width: int = None, height: int = None):

    output_path = get_unique_filename(output_path)
    layout = fig.layout
    fig.update_layout(dict1=dict(updatemenus=[], sliders=[]), overwrite=True)

    fig.layout.width = width or fig.layout.width or 1080
    fig.layout.height = height or fig.layout.height or 607
    print(f"Image size: {fig.layout.width}x{fig.layout.height}")

    num_frames = len(fig.frames)

    process = (
        ffmpeg.input("pipe:0", framerate=fps, format="image2pipe", pix_fmt="yuv420p")  # 通过stdin传输图像
        .output(
            output_path,
            vcodec="h264_nvenc",
            cq=19,
        )
        .global_args("-loglevel", "warning")  # 设置日志级别为 quiet
        .run_async(pipe_stdin=True)  # 异步运行，打开stdin管道
    )

    for i in tqdm.tqdm(range(num_frames)):
        fig.update(data=fig.frames[i].data)
        img_data = fig.to_image(format="png")
        process.stdin.write(img_data)

    # 关闭stdin，告诉ffmpeg输入已经完成
    process.stdin.close()

    process.wait()
    fig.layout = layout
    print(f"Video saved to {output_path}")

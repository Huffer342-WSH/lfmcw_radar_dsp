import tkinter as tk
import time
from scipy.io import savemat
import numpy as np
from tkinter import filedialog
from scipy.signal import savgol_filter


def find_duplicates(arr):
    seen = {}
    duplicates = []

    # 遍历数组并记录每个元素的位置
    for i, value in enumerate(arr):
        if value in seen:
            duplicates.append((value, seen[value], i))  # 记录重复元素及其索引
        else:
            seen[value] = i

    # 打印结果
    if duplicates:
        for value, first_index, second_index in duplicates:
            print(f"元素 {value} 在位置 {first_index} 和 {second_index} 重复")
    else:
        print("数组中没有重复元素")


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("鼠标轨迹录制")
        self.root.geometry("800x650")

        # 创建画布
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 创建保存按钮
        self.save_button = tk.Button(self.root, text="保存轨迹", command=self.save_data)
        self.save_button.pack(pady=10, side=tk.BOTTOM)  # 确保按钮在底部显示

        # 绘制坐标轴
        self.draw_axes()

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # 记录鼠标轨迹
        self.positions = []  # 鼠标位置数组
        self.timestamps = []  # 采样时刻数组
        self.start_time = None  # 起始时间
        self.drawing = False  # 标记是否在绘制

        self.timePrev = -1

    def draw_axes(self):
        """绘制坐标轴"""
        width = int(self.canvas["width"])
        height = int(self.canvas["height"])

        # X 轴朝上，Y 轴朝左
        self.canvas.create_line(400, height, 400, 0, fill="black", arrow=tk.LAST)  # X 轴
        self.canvas.create_line(800, height - 50, 0, height - 50, fill="black", arrow=tk.LAST)  # Y 轴

        # 在坐标轴上加标记
        self.canvas.create_text(410, 10, text="X", fill="black")  # X 轴朝上
        self.canvas.create_text(10, 290, text="Y", fill="black")  # Y 轴朝左

    def start_draw(self, event):
        """开始绘制，记录初始位置和时间"""
        self.drawing = True
        self.positions.append((event.x, event.y))
        if self.start_time is None:
            self.start_time = time.time()  # 记录初始时间
        self.timestamps.append(0)  # 初始时刻为 0

    def draw(self, event):
        """绘制过程中的鼠标位置和时间采集"""
        if self.drawing:
            timeNew = time.time() - self.start_time
            if timeNew == self.timePrev:
                self.timePrev = timeNew
                return
            self.positions.append((event.x, event.y))
            self.timestamps.append(timeNew)  # 记录相对时间
            # 绘制线段
            self.canvas.create_line(self.positions[-2], self.positions[-1], fill="black", width=2)
            self.timePrev = timeNew

    def stop_draw(self, event):
        """结束绘制，停止记录"""
        if self.drawing:
            self.drawing = False

    def save_data(self):
        """弹出文件管理器，保存数据到MATLAB格式的文件"""
        if self.positions and self.timestamps:
            # 转换为 NumPy 数组
            positions_np = savgol_filter(np.array(self.positions), 20, 2, axis=0)
            positions_new = np.zeros(shape=(len(positions_np), 3))
            positions_new[:, 0] = (550 - positions_np[:, 1]) / 300
            positions_new[:, 1] = (400 - positions_np[:, 0]) / 300

            timestamps_np = np.array(self.timestamps)
            find_duplicates(timestamps_np)

            # 获取当前时间，生成默认文件名
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
            default_filename = f"mouse_trajectory_{current_time}.mat"

            # 弹出文件管理器，选择保存文件名和路径，带默认文件名
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mat", filetypes=[("MATLAB files", "*.mat")], initialfile=default_filename, title="Save as"
            )
            if file_path:
                # 创建数据字典
                data = {"positions": positions_new, "timestamps": timestamps_np}

                # 保存到选定路径
                savemat(file_path, data, do_compression=True)
                print(f"Data saved to {file_path}")

                # 清空时间戳和轨迹，准备记录下一个轨迹
                self.positions = []
                self.timestamps = []
                self.start_time = None
                self.canvas.delete("all")  # 清空画布
                self.draw_axes()  # 重新绘制坐标轴


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

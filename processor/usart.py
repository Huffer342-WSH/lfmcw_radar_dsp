# %%
import serial
import serial.tools.list_ports


class Usart(serial.Serial):
    def __init__(self) -> None:
        self.search()
        self.portIndex = 0

    def search(self):
        self.portsList = list(serial.tools.list_ports.comports())

    def connect_by_CLI(self, baudrate=None):
        self.search()
        count = self.portsList.__len__()
        if count == 0:
            print("没有识别到串口，退出。")
            return
        for i in range(count):
            print(i, ":", self.portsList[i])

        while True:
            try:
                # 提示用户输入一个数字，并从控制台读取输入
                print("请输入 需要连接的串口序号：")
                input_str = input()
                # 尝试将输入转换为数字
                num = int(input_str)
                if num >= 0 and num < count:
                    self.portIndex = num
                    # 如果转换成功，跳出循环
                    break
                else:
                    print("输入的序号超出有效范围，请重新输入")
            except ValueError:
                # 如果转换失败，打印错误信息并继续循环
                print("输入的不是一个整数，请重新输入。")

        if baudrate is None:
            while True:
                try:
                    # 提示用户输入一个数字，并从控制台读取输入
                    print("请输入 串口使用的波特率：")
                    input_str = input()
                    # 尝试将输入转换为数字
                    baudrate = int(input_str)
                    break
                except ValueError:
                    # 如果转换失败，打印错误信息并继续循环
                    print("输入的不是一个整数，请重新输入。")

        serial.Serial.__init__(self, port=self.portsList[self.portIndex].device, baudrate=baudrate)

    def connect(self, index=0, baudrate=115200):
        if self.portsList.__len__() > index:
            super(Usart, self).__init__(self.portsList[index].device, baudrate)
            return True
        else:
            return False

    def connect_by_name(self, name_part, baudrate=115200):
        self.search()
        for port in self.portsList:
            if name_part in port.device or name_part in port.description:
                try:
                    super(Usart, self).__init__(port.device, baudrate)
                    print(f"已连接到: {port.device} ({port.description})")
                    return True
                except serial.SerialException:
                    print(f"无法连接到: {port.device} ({port.description})")
                return False
        print(f"未找到包含 '{name_part}' 的串口设备。")
        return False


def list_and_select_serial_port():
    ports = serial.tools.list_ports.comports()
    port_list = []

    if not ports:
        print("No serial ports found.")
        return None

    print("Available serial ports:")
    for i, port in enumerate(ports):
        port_info = f"{i + 1}: {port.device} - {port.description}"
        print(port_info)
        port_list.append(port.device)

    while True:
        try:
            choice = int(input("Select a port by number: ")) - 1
            if 0 <= choice < len(port_list):
                selected_port = port_list[choice]
                print(f"You selected: {selected_port}")
                return selected_port
            else:
                print("Invalid choice, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")


# %%

if __name__ == "__main__":

    usart = Usart()
    # usart.connect_by_CLI()
    # usart.connect(index=3, baudrate=2000000)
    usart.connect_by_name("COM9", 2000000)

    while True:
        while usart.read(1) != b"\x55":
            pass
        usart.read(1)
        while True:
            data = usart.read(520)
            if data[1] != 0xAA:
                print(data[1])
                break
            print(data)

import multiprocessing

import serial
import serial.tools.list_ports
import time


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = multiprocessing.Array("c", size)  # Shared memory array
        self.start = multiprocessing.Value("i", 0)  # Start of the buffer
        self.end = multiprocessing.Value("i", 0)  # End of the buffer
        self.lock = multiprocessing.Lock()  # Ensure thread-safe access
        self.data_available = multiprocessing.Condition()

    def write(self, data):
        with self.lock:
            data_len = len(data)
            space_left = self.size - self._used_space()
            if data_len > space_left:
                data = data[-space_left:]  # Only keep the last part that fits
                data_len = len(data)

            write_end = (self.end.value + data_len) % self.size

            if self.end.value + data_len <= self.size:
                self.buffer[self.end.value : self.end.value + data_len] = data
            else:
                first_part = self.size - self.end.value
                self.buffer[self.end.value :] = data[:first_part]
                self.buffer[:write_end] = data[first_part:]

            self.end.value = write_end

            if self._used_space() > self.size:
                self.start.value = (self.start.value + data_len) % self.size

        with self.data_available:
            self.data_available.notify_all()

    def read(self, size):
        with self.lock:
            available_data = self._used_space()
            size = min(size, available_data)

            read_end = (self.start.value + size) % self.size
            if self.start.value + size <= self.size:
                data = self.buffer[self.start.value : self.start.value + size]
            else:
                first_part = self.size - self.start.value
                data = self.buffer[self.start.value :] + self.buffer[:read_end]

            self.start.value = read_end
            return bytes(data)

    def _used_space(self):
        if self.end.value >= self.start.value:
            return self.end.value - self.start.value
        else:
            return self.size - self.start.value + self.end.value


class Usart(multiprocessing.Process):
    def __init__(self, port, baudrate, buffer_size=None):
        super().__init__()
        if buffer_size is None:
            buffer_size = baudrate * 2
        self.port = port
        self.baudrate = baudrate
        self.buffer = CircularBuffer(buffer_size)
        self.serial_connection = serial.Serial()
        self.serial_connection.port = self.port
        self.serial_connection.baudrate = self.baudrate
        self._stop_event = multiprocessing.Event()

    def run(self):
        """Main loop to continuously read data from serial port."""
        self.serial_connection.open()
        while not self._stop_event.is_set():
            if self.serial_connection.in_waiting:
                data = self.serial_connection.read(self.serial_connection.in_waiting)
                self.buffer.write(data)
        self.serial_connection.close()

    def stop(self):
        """Signal the process to stop."""
        self._stop_event.set()

    def read(self, size=None, timeout=None):
        """Advanced read function. Blocks until the specified size is read or returns all available data if size is None."""
        data = bytearray()
        start_time = time.time()
        with self.buffer.data_available:
            while size is None or len(data) < size:
                chunk = self.buffer.read(size - len(data) if size else self.buffer._used_space())
                data.extend(chunk)

                if size is None or len(data) == size:
                    break
                if size and len(data) > size:
                    raise ValueError("Requested size is larger than available data.")

                if timeout != None and time.time() - start_time > timeout:
                    break

                self.buffer.data_available.wait()

        return bytes(data)

    def in_waiting(self):
        return self.buffer._used_space()

    def select_serial_port():
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


if __name__ == "__main__":
    # Example usage
    usart = Usart(port="/dev/ttyUSB2", baudrate=3250000, buffer_size=2048)
    usart.start()

    try:
        while True:
            data = usart.read(100)  # Blocking read for 100 bytes
            if data:
                print(f"Received: {data}")
                print(f"Buffer size: {usart.buffer._used_space()}")
                print(f"In waiting: {usart.in_waiting()}")
    except KeyboardInterrupt:
        usart.stop()
        usart.join()

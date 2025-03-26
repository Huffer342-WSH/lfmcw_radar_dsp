import numpy as np


import numpy as np


def numpy_to_c_array(array, array_name, c_type):
    """
    将任意维度的NumPy数组转换为C语言数组格式。

    参数：
        array (numpy.ndarray): 要转换的NumPy数组。
        array_name (str): C语言中数组的名称。
        c_type (str): C语言的数据类型（如int, float, double等）。

    返回：
        str: 格式化的C语言数组字符串。
    """

    def format_array(arr, indent_level=0):
        indent = "    " * indent_level  # 缩进字符串，每层增加4个空格
        # 如果是标量，则直接返回字符串表示
        if arr.ndim == 0:
            return str(arr)
        # 如果是一维数组，格式化为C语言的数组风格
        if arr.ndim == 1:
            return indent + "{" + ", ".join(map(str, arr)) + "}"
        # 如果是多维数组，递归处理，每层增加缩进
        inner = ",\n".join(format_array(sub_arr, indent_level + 1) for sub_arr in arr)
        return indent + "{" + "\n" + inner + "\n" + indent + "}"

    type_map = {
        "float": np.float32,
        "double": np.float64,
        "int64_t": np.int64,
        "int32_t": np.int32,
        "int16_t": np.int16,
        "int8_t": np.int8,
        "uint64_t": np.uint64,
        "uint32_t": np.uint32,
        "uint16_t": np.uint16,
        "uint8_t": np.uint8,
    }

    if c_type not in type_map:
        raise ValueError(f"Unsupported C type: {c_type}")

    numpy_type = type_map[c_type]
    array = array.astype(numpy_type)

    # 获取数组的形状，并转换为C风格的声明
    shape_str = "][".join(map(str, array.shape))
    c_array = f"{c_type} {array_name}[{shape_str}] = {format_array(array)};"

    return c_array


# 示例
if __name__ == "__main__":
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    c_code = numpy_to_c_array(arr, "my_array", "int32_t")
    print(c_code)

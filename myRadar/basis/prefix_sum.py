import numpy as np


class PrefixSum2D:
    def __init__(self, mat):
        """
        根据输入的二维矩阵，初始化类，并计算前缀和

        参数:
        - mat: 输入的二维数组
        """
        self.mat = mat
        self.prefix_sum = self.__compute_prefix_sum(mat)

    def __compute_prefix_sum(self, mat):
        """
        使用numpy.cumsum计算二维矩阵的前缀和，并在前缀和矩阵的左上角添加一行和一列的零

        参数:
        - mat: 输入的二维数组

        返回:
        - prefix_sum: 一个二维数组，其中每个元素是从左上角到当前位置的元素之和，第一行和第一列为额外停驾的0

        """
        # Get the cumulative sum along both axes, then pad with an extra row and column of zeros at the top and left
        prefix_sum = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1), dtype=np.float64)
        prefix_sum[1:, 1:] = np.cumsum(np.cumsum(mat, axis=0), axis=1)

        return prefix_sum

    def getSum(self, x1, y1, x2, y2):
        """
        使用前缀和数组，计算从(x1, y1)到(x2, y2)的子矩阵的和

        参数:
        - (x1, y1): 子矩阵的左上角角点(闭区间)
        - (x2, y2): 子矩阵的右下角角点(闭区间)

        返回:
        - 指定区域内的所有元素的总和
        """
        total = self.prefix_sum[x2 + 1, y2 + 1]  # Adjust indices for the extra row and column
        total -= self.prefix_sum[x1, y2 + 1]
        total -= self.prefix_sum[x2 + 1, y1]
        total += self.prefix_sum[x1, y1]
        return total


if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    ps = PrefixSum2D(mat)

    # 获取子矩阵 (0, 0) 到 (1, 1) 的和
    area_sum = ps.getSum(0, 0, 1, 1)
    print(area_sum)  # 输出应该是12（1 + 2 + 4 + 5）

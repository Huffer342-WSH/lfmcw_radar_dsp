import numpy as np
from math import sqrt
from abc import abstractmethod


class KalmanModel:
    @abstractmethod
    def function(self, state_vector, **kwargs):
        pass

    @abstractmethod
    def covar(self, **kwargs):
        pass

    @abstractmethod
    def matirx(self, **kwargs):
        pass


class TransitionModel(KalmanModel):
    def __init__(self, velocity_noise_coef):
        self.velocity_noise_coef = velocity_noise_coef

    def function(self, state_vector, delta_time):
        F = self.matirx(delta_time)
        x_pred = F @ state_vector
        return x_pred

    def matirx(self, delta_time):
        cell_F = np.array([[1, delta_time], [0, 1]])
        F = np.kron(np.eye(3), cell_F)
        return F

    def covar(self, delta_time):
        cell_Q = np.array([[delta_time**3 / 3, delta_time**2 / 2], [delta_time**2 / 2, delta_time]])
        cell_Q *= self.velocity_noise_coef
        Q = np.kron(np.eye(3), cell_Q)
        return Q


class MeasurementModel(KalmanModel):
    def __init__(self, meas_noise_covar):
        self.meas_noise_covar = meas_noise_covar

    def function(self, state_vector):
        xyz_pos = state_vector[[0, 2, 4], :]
        xyz_vel = state_vector[[1, 3, 5], :]
        x, y, z = xyz_pos
        rho = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arcsin(z / rho)

        xyz_pos = np.vstack((x, y, z))
        rho_rate = np.einsum("ij,ij->j", xyz_pos, xyz_vel) / np.linalg.norm(xyz_pos, axis=0)

        return np.vstack((theta, phi, rho, rho_rate))

    def matrix(self, state_vector):
        """返回测量矩阵 H ， 扩展卡尔曼计算雅可比矩阵
        计算雅可比矩阵有两种方法
        - 数学方法得到求导后的公式带入参数计算
        - 使用两个状态向量x和x+dx 带入measurement_function(x), 实现近似求导
        """
        jac = np.zeros((4, 6), dtype=np.float64)
        x, vx, y, vy, z, vz = state_vector
        x2, y2, z2 = x**2, y**2, z**2
        x2y2 = x2 + y2
        r2 = x2y2 + z2
        r = sqrt(r2)
        sqrt_x2_y2 = sqrt(x2y2)
        r32 = r2 * r

        # 测量向量   [theta, phi, r, rdot]'
        # 状态向量   [x, vx, y, vy, z, vz]'

        # dtheta/dx
        sqrt_x2_y2r2 = sqrt_x2_y2 * r2
        jac[0, 0] = -(x * z) / (sqrt_x2_y2r2)

        # dtheta/dy
        jac[0, 2] = -(y * z) / (sqrt_x2_y2r2)

        # dthtea/dz
        jac[0, 4] = sqrt_x2_y2 / r2

        # dphi/dx
        jac[1, 0] = -y / (x2y2)

        # dphi/dy
        jac[1, 2] = x / (x2y2)

        # dphi/dz = 0

        # dr/dx and drdot/dvx
        jac[2, 0] = jac[3, 1] = x / r

        # dr/dx and drdot/dvy
        jac[2, 2] = jac[3, 3] = y / r

        # dr/dx and drdot/dvz
        jac[2, 4] = jac[3, 5] = z / r

        vx_x, vy_y, vz_z = vx * x, vy * y, vz * z

        # drdot/dx
        jac[3, 0] = (-x * (vy_y + vz_z) + vx * (y2 + z2)) / r32

        # drdot/dy
        jac[3, 2] = (vy * (x2 + z2) - y * (vx_x + vz_z)) / r32

        # drdot/dz
        jac[3, 4] = (vz * (x2y2) - (vx_x + vy_y) * z) / r32
        return jac

    def covar(self, measurement=None):
        return self.meas_noise_covar

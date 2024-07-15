class ConfigParameters:
    def __init__(self):
        self.earth_rotation_speed_ = 7.2921159e-5  # 地球自转速度
        self.earth_gravity_ = 9.80665  # 重力加速度

        self.position_error_prior_std_ = 1.0  # 初始位置误差标准差
        self.velocity_error_prior_std_ = 1.0  # 初始速度误差标准差
        self.rotation_error_prior_std_ = 0.1  # 初始旋转误差标准差
        self.gyro_bias_error_prior_std_ = 0.01  # 初始陀螺仪偏差误差标准差
        self.accelerometer_bias_error_prior_std_ = 0.01  # 初始加速度计偏差误差标准差

        self.gyro_noise_std_ = 0.005  # 陀螺仪噪声标准差
        self.accelerometer_noise_std_ = 0.05  # 加速度计噪声标准差

        self.adns_position_x_std_ = 0.001  # ADNS测量噪声在X方向上的标准差
        self.adns_position_y_std_ = 0.001  # ADNS测量噪声在Y方向上的标准差

        self.use_earth_model_ = False  # 是否使用地球模型


import pandas as pd
from datetime import datetime

class IMUData:
    def __init__(self, linear_accel, angle_velocity, time):
        self.linear_accel = linear_accel
        self.angle_velocity = angle_velocity
        self.time = time

def read_imu_data(file_path):
    imu_data_list = []
    df = pd.read_csv('data__1.csv', encoding='utf-8', sep=',', skipinitialspace=True, index_col=False)
    for _, row in df.iterrows():
        time = datetime.strptime(row['Time'], '%H:%M:%S.%f')
        linear_accel = [row['AccelX_g'], row['AccelY_g'], row['AccelZ_g']]
        angle_velocity = [row['GyroX_deg_per_s'], row['GyroY_deg_per_s'], row['GyroZ_deg_per_s']]
        imu_data_list.append(IMUData(linear_accel, angle_velocity, time))
    return imu_data_list


class ADNSData:
    def __init__(self, local_position_ned, time, cpi):
        self.local_position_ned = np.array(local_position_ned) * 0.0254 / cpi
        self.time = time

def read_adns_data(file_path, cpi):
    adns_data_list = []
    df = pd.read_csv('adns.csv', encoding='utf-8', sep=',', skipinitialspace=True, index_col=False)
    for _, row in df.iterrows():
        time = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
        local_position_ned = [row['xvalue'], row['yvalue'], 0.0]  # 假设z值为0
        adns_data_list.append(ADNSData(local_position_ned, time, cpi))
    return adns_data_list


import numpy as np
from scipy.linalg import expm

class ErrorStateKalmanFilter:
    def __init__(self, config_parameters):
        self.config_parameters_ = config_parameters
        self.earth_rotation_speed_ = config_parameters.earth_rotation_speed_
        self.g_ = np.array([0.0, 0.0, -config_parameters.earth_gravity_])
        self.X_ = np.zeros((15, 1))
        self.F_ = np.zeros((15, 15))
        self.C_ = np.eye(3)
        self.G_ = np.zeros((3, 15))
        self.G_[:, :3] = np.eye(3)
        self.SetCovarianceP(config_parameters)
        self.SetCovarianceQ(config_parameters)
        self.SetCovarianceR(config_parameters)
        self.ResetState()

        self.gyro_bias_ = np.zeros(3)
        self.accel_bias_ = np.zeros(3)
        self.velocity_ = np.zeros(3)
        self.pose_ = np.eye(4)
        self.imu_data_buff_ = []
        self.curr_adns_data_ = None

        self.positions = []

    def SetCovarianceP(self, config):
        self.P_ = np.zeros((15, 15))
        self.P_[:3, :3] = np.eye(3) * config.position_error_prior_std_ ** 2
        self.P_[3:6, 3:6] = np.eye(3) * config.velocity_error_prior_std_ ** 2
        self.P_[6:9, 6:9] = np.eye(3) * config.rotation_error_prior_std_ ** 2
        self.P_[9:12, 9:12] = np.eye(3) * config.gyro_bias_error_prior_std_ ** 2
        self.P_[12:15, 12:15] = np.eye(3) * config.accelerometer_bias_error_prior_std_ ** 2

    def SetCovarianceQ(self, config):
        self.Q_ = np.zeros((6, 6))
        self.Q_[:3, :3] = np.eye(3) * config.gyro_noise_std_ ** 2
        self.Q_[3:6, 3:6] = np.eye(3) * config.accelerometer_noise_std_ ** 2

    def SetCovarianceR(self, config):
        self.R_ = np.zeros((3, 3))
        self.R_[0, 0] = config.adns_position_x_std_ ** 2  # ADNS测量噪声在X方向上的标准差
        self.R_[1, 1] = config.adns_position_y_std_ ** 2  # ADNS测量噪声在Y方向上的标准差
        self.R_[2, 2] = 1e-6  # 假设Z方向上的噪声极小

    def Init(self, curr_adns_data, curr_imu_data):
        self.velocity_ = np.zeros(3)
        self.pose_ = np.eye(4)
        self.pose_[:3, 3] = curr_adns_data.local_position_ned
        self.imu_data_buff_ = [curr_imu_data]
        self.curr_adns_data_ = curr_adns_data
        self.positions.append(self.pose_[:3, 3].copy())
        return True

    def Predict(self, curr_imu_data):
        self.imu_data_buff_.append(curr_imu_data)
        delta_t = (curr_imu_data.time - self.imu_data_buff_[-2].time).total_seconds()
        w_in = np.array(curr_imu_data.angle_velocity)  # 确保 w_in 是 NumPy 数组
        self.UpdateOdomEstimation(w_in, delta_t, curr_imu_data)
        self.UpdateErrorState(delta_t, curr_imu_data.linear_accel, w_in)
        self.imu_data_buff_.pop(0)
        self.positions.append(self.pose_[:3, 3].copy() * 100)  # 将位置从米转换为厘米
        return True


    def Correct(self, curr_adns_data):
        self.curr_adns_data_ = curr_adns_data
        Y = curr_adns_data.local_position_ned - self.pose_[:3, 3]
        K = self.P_ @ self.G_.T @ np.linalg.inv(self.G_ @ self.P_ @ self.G_.T + self.C_ @ self.R_ @ self.C_.T)
        self.X_ = self.X_ + K @ (Y[:, np.newaxis] - self.G_ @ self.X_)
        self.P_ = (np.eye(15) - K @ self.G_) @ self.P_
        self.EliminateError()
        self.ResetState()
        self.positions.append(self.pose_[:3, 3].copy() * 100)  # 将位置从米转换为厘米
        return True


    def ComputeNavigationFrameAngularVelocity(self):
        return np.zeros(3)  # 省略实际计算，仅返回零向量

    def UpdateOdomEstimation(self, w_in, delta_t, curr_imu_data):
        w_in = np.array(w_in)  # 确保 w_in 是 NumPy 数组
        last_imu_data = self.imu_data_buff_[-2]
        delta_rotation = self.ComputeDeltaRotation(last_imu_data, curr_imu_data)
        R_nm_nm_1 = expm(self.BuildSkewSymmetricMatrix(w_in * delta_t)).T
        curr_R, last_R = self.pose_[:3, :3], self.pose_[:3, :3]
        self.ComputeOrientation(delta_rotation, R_nm_nm_1, curr_R, last_R)
        last_vel, curr_vel = self.velocity_, self.velocity_
        self.ComputeVelocity(last_R, curr_R, last_imu_data, curr_imu_data, last_vel, curr_vel)
        self.ComputePosition(last_R, curr_R, last_vel, curr_vel, last_imu_data, curr_imu_data)

    def UpdateErrorState(self, delta_t, accel, w_in):
        F_23 = self.BuildSkewSymmetricMatrix(accel)
        F_33 = -self.BuildSkewSymmetricMatrix(w_in)
        self.F_[:3, 3:6] = np.eye(3) * delta_t
        self.F_[3:6, 6:9] = -self.pose_[:3, :3] @ F_23 * delta_t
        self.F_[3:6, 9:12] = -self.pose_[:3, :3] * delta_t
        self.F_[6:9, 6:9] = np.eye(3) + F_33 * delta_t
        self.F_[6:9, 9:12] = -np.eye(3) * delta_t

        # 扩展 self.Q_ 的维度使其与 self.P_ 的维度一致
        Q_extended = np.zeros((15, 15))
        Q_extended[:6, :6] = self.Q_

        self.P_ = self.F_ @ self.P_ @ self.F_.T + Q_extended

    def EliminateError(self):
        self.pose_[:3, 3] += self.X_[:3, 0]
        self.velocity_ += self.X_[3:6, 0]
        self.pose_[:3, :3] = expm(self.BuildSkewSymmetricMatrix(-self.X_[6:9, 0])) @ self.pose_[:3, :3]
        self.gyro_bias_ += self.X_[9:12, 0]
        self.accel_bias_ += self.X_[12:15, 0]

    def ResetState(self):
        self.X_ = np.zeros((15, 1))

    def BuildSkewSymmetricMatrix(self, vec):
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

    def ComputeDeltaRotation(self, last_imu_data, curr_imu_data):
        # 示例，仅返回零矩阵
        return np.zeros((3, 3))

    def ComputeOrientation(self, delta_rotation, R_nm_nm_1, curr_R, last_R):
        curr_R = last_R @ R_nm_nm_1 @ delta_rotation.T

    def ComputeVelocity(self, last_R, curr_R, last_imu_data, curr_imu_data, last_vel, curr_vel):
        unbias_accel_0 = last_R @ (last_imu_data.linear_accel - self.accel_bias_) + self.g_
        unbias_accel_1 = curr_R @ (curr_imu_data.linear_accel - self.accel_bias_) + self.g_
        delta_t = (curr_imu_data.time - last_imu_data.time).total_seconds()
        self.velocity_ += 0.5 * delta_t * (unbias_accel_0 + unbias_accel_1)

    def ComputePosition(self, last_R, curr_R, last_vel, curr_vel, last_imu_data, curr_imu_data):
        unbias_accel_0 = last_R @ (last_imu_data.linear_accel - self.accel_bias_) + self.g_
        unbias_accel_1 = curr_R @ (curr_imu_data.linear_accel - self.accel_bias_) + self.g_
        delta_t = (curr_imu_data.time - curr_imu_data.time).total_seconds()
        self.pose_[:3, 3] += 0.5 * delta_t * (curr_vel + last_vel) + 0.25 * (unbias_accel_0 + unbias_accel_1) * delta_t ** 2






# 配置参数
config_parameters = ConfigParameters()

# 调整ADNS测量噪声标准差
config_parameters.adns_position_x_std_ = 0.001  # 调整为合理的噪声标准差
config_parameters.adns_position_y_std_ = 0.001  # 调整为合理的噪声标准差

config_parameters.accelerometer_noise_std_ = 0.05  # 减小加速度计噪声标准差
config_parameters.gyro_noise_std_ = 0.005  # 减小陀螺仪噪声标准差

# 创建滤波器实例
eskf = ErrorStateKalmanFilter(config_parameters)

# 读取IMU和ADNS数据
imu_data_list = read_imu_data('/mnt/data/imu_data.csv')
adns_data_list = read_adns_data('/mnt/data/adns_data.csv', 8200)

# 确认数据类型和内容
print(f"Type of first ADNSData object: {type(adns_data_list[0])}")
print(f"Contents of first ADNSData object: {adns_data_list[0].__dict__}")
print(f"Type of first IMUData object: {type(imu_data_list[0])}")
print(f"Contents of first IMUData object: {imu_data_list[0].__dict__}")

# 初始化滤波器
eskf.Init(adns_data_list[0], imu_data_list[0])  # 确保传递的是ADNSData对象和IMUData对象

# 运行滤波器
for imu_data, adns_data in zip(imu_data_list[1:], adns_data_list[1:]):
    eskf.Predict(imu_data)
    eskf.Correct(adns_data)

# 获取记录的轨迹
positions = np.array(eskf.positions)

# 绘制轨迹
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1], label='Trajectory')
plt.scatter(positions[-1, 0], positions[-1, 1], color='red', marker='o')  # 标记终点
plt.text(positions[-1, 0], positions[-1, 1], 'End', fontsize=12, color='red')  # 添加注释
plt.xlabel('X Position (cm)')
plt.ylabel('Y Position (cm)')
plt.title('Trajectory of the IMU+ADNS Data Fusion')
plt.legend()
plt.grid(True)
plt.show()

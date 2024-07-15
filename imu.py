import pandas as pd
from datetime import datetime

class IMUData:
    def __init__(self, linear_accel, angle_velocity, time):
        self.linear_accel = np.array(linear_accel)
        self.angle_velocity = np.array(angle_velocity)
        self.time = time

def read_imu_data(file_path):
    imu_data_list = []
    df = pd.read_csv('imu.csv', encoding='utf-8', sep=',', skipinitialspace=True, index_col=False)
    for _, row in df.iterrows():
        time = datetime.strptime(row['Time'], '%H:%M:%S.%f')
        # 将加速度从g转换为cm/s²，其他代码保持不变
        # 将加速度从g转换为cm/s²
        linear_accel = [row['AccelX_g'] * 980.665, row['AccelY_g'] * 980.665, row['AccelZ_g'] * 980.665]


        angle_velocity = [row['GyroX_deg_per_s'], row['GyroY_deg_per_s'], row['GyroZ_deg_per_s']]
        imu_data_list.append(IMUData(linear_accel, angle_velocity, time))
    return imu_data_list


import numpy as np

class SimpleIMUTracker:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.positions = []

    def update(self, imu_data, delta_t):
        # 更新速度
        accel = np.array(imu_data.linear_accel)  # 单位已经是cm/s²
        self.velocity += accel * delta_t
        # 更新位置
        self.position += self.velocity * delta_t
        # 记录位置
        self.positions.append(self.position.copy())

def compute_trajectory(imu_data_list):
    tracker = SimpleIMUTracker()
    for i in range(1, len(imu_data_list)):
        delta_t = (imu_data_list[i].time - imu_data_list[i - 1].time).total_seconds()
        tracker.update(imu_data_list[i], delta_t)
    return np.array(tracker.positions)


import matplotlib.pyplot as plt

def plot_trajectory(positions):
    plt.figure(figsize=(10, 6))
    plt.plot(positions[:, 0], positions[:, 1], label='Trajectory')
    plt.scatter(positions[-1, 0], positions[-1, 1], color='red', marker='o')  # 标记终点
    plt.text(positions[-1, 0], positions[-1, 1], 'End', fontsize=12, color='red')  # 添加注释
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('Trajectory using IMU Data')
    plt.legend()
    plt.grid(True)
    plt.show()

# 读取IMU数据
imu_data_list = read_imu_data('/mnt/data/imu_data.csv')

# 计算轨迹
positions = compute_trajectory(imu_data_list)

# 绘制轨迹
plot_trajectory(positions)

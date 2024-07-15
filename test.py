import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据并设置index_col=False
data_path = 'data__1.csv'

# 确认使用逗号作为分隔符，并删除多余的空白字符
try:
    data = pd.read_csv(data_path, encoding='utf-8', sep=',', skipinitialspace=True, index_col=False)
except UnicodeDecodeError:
    data = pd.read_csv(data_path, encoding='gbk', sep=',', skipinitialspace=True, index_col=False)

# 重命名列
data.columns = ["Time", "DeviceName", "AccelX_g", "AccelY_g", "AccelZ_g", "GyroX_deg_per_s", "GyroY_deg_per_s", "GyroZ_deg_per_s", "AngleX_deg", "AngleY_deg", "AngleZ_deg", "MagX_uT", "MagY_uT", "MagZ_uT"]

# 将时间字符串转换为秒数
def time_to_seconds(t):
    try:
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return np.nan

# 转换时间列并检查无效数据
data['Time_sec'] = data['Time'].apply(time_to_seconds)
time = data['Time_sec'].values

# 提取IMU数据
accel_x = pd.to_numeric(data['AccelX_g'].values, errors='coerce') * 9.80665  # 转换为 m/s^2
accel_y = pd.to_numeric(data['AccelY_g'].values, errors='coerce') * 9.80665  # 转换为 m/s^2
accel_z = pd.to_numeric(data['AccelZ_g'].values, errors='coerce') * 9.80665  # 转换为 m/s^2
gyro_x = pd.to_numeric(data['GyroX_deg_per_s'].values, errors='coerce')
gyro_y = pd.to_numeric(data['GyroY_deg_per_s'].values, errors='coerce')
gyro_z = pd.to_numeric(data['GyroZ_deg_per_s'].values, errors='coerce')
mag_x = pd.to_numeric(data['MagX_uT'].values, errors='coerce')
mag_y = pd.to_numeric(data['MagY_uT'].values, errors='coerce')
mag_z = pd.to_numeric(data['MagZ_uT'].values, errors='coerce')

# 绘制结果
plt.figure(figsize=(10, 12))

# 陀螺仪数据
plt.subplot(3, 1, 1)
plt.plot(time, gyro_x, label='X')
plt.plot(time, gyro_y, label='Y')
plt.plot(time, gyro_z, label='Z')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
plt.legend()
plt.title('Gyroscope')

# 加速度计数据
plt.subplot(3, 1, 2)
plt.plot(time, accel_x, label='X')
plt.plot(time, accel_y, label='Y')
plt.plot(time, accel_z, label='Z')
plt.xlabel('Time (s)')
plt.ylabel('m/s^2')
plt.legend()
plt.title('Accelerometer')

# 磁力计数据
plt.subplot(3, 1, 3)
plt.plot(time, mag_x, label='X')
plt.plot(time, mag_y, label='Y')
plt.plot(time, mag_z, label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Gauss')
plt.legend()
plt.title('Magnetometer')

plt.tight_layout()
plt.show()

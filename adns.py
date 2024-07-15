import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 假设cpi为3400
cpi = 3400
conversion_factor = 2.54 / cpi  # 每个count对应的厘米

# 读取CSV文件并指定列名
data = pd.read_csv('adns.csv')
print(data)
# 计算每个时间间隔内的位移
def calculate_displacement(data):
    distances = [0]  # 初始距离为0
    total_distance = 0
    
    for i in range(1, len(data)):
        delta_x = (data['xvalue'][i] - data['xvalue'][i - 1])* conversion_factor
        delta_y = (data['yvalue'][i] - data['yvalue'][i - 1])* conversion_factor
        distance = np.sqrt(delta_x**2 + delta_y**2)
        total_distance += distance
        distances.append(total_distance)
    
    return distances

# 解析时间戳
def parse_timestamps(timestamps):
    times = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S.%f')
    time_deltas = (times - times.iloc[0]).dt.total_seconds()
    return time_deltas

# 计算位移
distances = calculate_displacement(data)

# 解析时间
times = parse_timestamps(data['time'])

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(times, distances, color='black', linestyle='-', marker='o', label='Traveled Distance')
plt.title('Traveled Distance over Time')
plt.xlabel('Time (s)')
plt.ylabel('Traveled Distance (cm)')
plt.grid(True)
plt.legend()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('adns.csv')

# 初始化累积位置
initial_position = (0, 0)
trajectory = [initial_position]
current_position = list(initial_position)
timestamps = [df.iloc[0]['time']]

# 计算累积位置并进行单位转换（假设原始数据单位是毫米，转换为厘米）
for index, row in df.iterrows():
    if index == 0:
        continue  # 跳过初始位置
    
    # 顺时针旋转90度坐标转换
    delta_x = row['yvalue'] * 25.4/8200 # 将y值作为新的x值，并转换为厘米
    delta_y = -row['xvalue']* 25.4/8200   # 将x值作为新的y值，并转换为厘米
    
    current_position[0] += delta_x
    current_position[1] += delta_y
    trajectory.append(tuple(current_position))
    timestamps.append(row['time'])

# 分离 X 和 Y 坐标
Delta_X, Delta_Y = zip(*trajectory)

# 创建图形
plt.figure()

# 绘制点和轨迹
plt.plot(Delta_X, Delta_Y, 'ro-')  # 'ro-'表示红色圆点并连接线

# 设置标题和标签
plt.title('Trajectory Plot from Incremental Data (cm, rotated 90°)')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')

# 显示图形
plt.grid(True)
plt.show()

# 保存累积位置数据，包含时间戳
df_trajectory = pd.DataFrame({'x': Delta_X, 'y': Delta_Y, 'time': timestamps})
df_trajectory.to_csv('cumulative_trajectory_cm_rotated.csv', index=False)

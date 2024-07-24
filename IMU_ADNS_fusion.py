import numpy as np
from datetime import datetime
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 处理 ADNS 数据
# ==============

# 文件路径
file_path = 'adns.txt'

# 创建空列表来存储数据
xvalues = []
yvalues = []
timestamps = []

# 从文件中读取数据
with open(file_path, 'r') as file:
    # 跳过标题行
    next(file)

    # 解析每一行数据
    for line in file:
        print( line.strip().split(','))
        x, y, t = line.strip().split(',')
        xvalues.append(float(x))
        yvalues.append(float(y))
        timestamps.append(datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))

# 将列表转换为NumPy数组
xvalues_array = np.array(xvalues)
yvalues_array = np.array(yvalues)
timestamps_array = np.array(timestamps)

# 计算位移
displacement = np.sqrt(np.square(xvalues_array) + np.square(yvalues_array))

# 创建ADNS DataFrame
df_adns = pd.DataFrame({
    'Timestamp': timestamps_array,
    'Displacement': displacement
})

# 将Timestamp设置为索引
df_adns.set_index('Timestamp', inplace=True)

print("\nADNS DataFrame head:")
print(df_adns.head())

# 处理 IMU 数据
# =============

# 定义IMU数据文件的路径
file_path = 'imu.txt'

# 读取IMU数据
data = []
with open(file_path, 'r') as file:
    # 读取并保存列名
    column_names = file.readline().strip().split(',')
    
    for line in file:
        # 去除行尾的换行符，然后分割每个值
        values = line.strip().split(',')
        # 将时间保持为字符串，其他值转换为浮点数，处理空值
        row = [values[0]] + [float(v) if v else np.nan for v in values[1:]]
        data.append(row)

# 将数据转换为 numpy 数组
data_array = np.array(data)

# 找到 AngleX_deg, AngleY_deg, AngleZ_deg 的列索引
angle_columns = ['AngleX_deg', 'AngleY_deg', 'AngleZ_deg']
angle_indices = [column_names.index(col) for col in angle_columns]

# 提取时间列和角度数据
time_column = data_array[:, 0]
angle_data = data_array[:, angle_indices]

# 创建 IMU DataFrame
df_imu = pd.DataFrame(angle_data, columns=angle_columns, index=time_column)

# 将索引列命名为 'Time'
df_imu.index.name = 'Time'

# 将数据类型转换为 float
df_imu = df_imu.astype(float)

print("\nIMU DataFrame head:")
print(df_imu.head())

# 数据预处理和合并
# ================

# 确保 df_adns 的索引是 datetime 类型
df_adns.index = pd.to_datetime(df_adns.index)

# 从 IMU 数据中提取小时和分钟
imu_time = pd.to_datetime(df_imu.index[0])
hour = imu_time.hour
minute = imu_time.minute

# 调整 ADNS 的时间
def adjust_time(timestamp):
    return timestamp.replace(hour=hour, minute=minute)

df_adns.index = df_adns.index.map(adjust_time)

# 统一时间格式：对于 df_imu，需要添加日期部分
date_part = df_adns.index[0].strftime('%Y-%m-%d ')
df_imu.index = pd.to_datetime(date_part + df_imu.index.astype(str))

# 确保两个 DataFrame 的索引都是 datetime 类型，并且是 timezone-naive
df_adns.index = pd.to_datetime(df_adns.index).tz_localize(None)
df_imu.index = pd.to_datetime(df_imu.index).tz_localize(None)

# 检查时间范围是否重叠
if df_adns.index.max() < df_imu.index.min() or df_imu.index.max() < df_adns.index.min():
    raise ValueError("The time ranges of df_adns and df_imu do not overlap.")

# 找到共同的时间范围
start_time = max(df_adns.index.min(), df_imu.index.min())
end_time = min(df_adns.index.max(), df_imu.index.max())
print(start_time,end_time)
# 创建一个统一的时间索引（使用 0.1s 频率）
common_index = pd.date_range(start=pd.to_datetime(start_time), end=pd.to_datetime(end_time), freq='100L')

# 对 ADNS DataFrame 使用 NumPy 进行插值
original_times = (df_adns.index - df_adns.index[0]).total_seconds().values
new_times = (common_index - common_index[0]).total_seconds().values
interpolated_values = np.interp(new_times, original_times, df_adns['Displacement'].values)
df_adns_resampled = pd.DataFrame({'Displacement': interpolated_values}, index=common_index)

# 对 IMU DataFrame 进行重采样和插值
df_imu_resampled = df_imu.reindex(common_index).interpolate(method='time')
print("\n----------------------------------------------:")
print(common_index)
print(df_imu_resampled)

# 合并两个重采样后的 DataFrame
df_combined = pd.concat([df_adns_resampled, df_imu_resampled], axis=1)

# 打印结果
print("\n合并后的 DataFrame 尾部:")
print(df_combined.tail(15))

# 数据可视化
# ==========

# 计算 x, y, z 的位置
df_combined['x'] = df_combined['Displacement'] * np.sin(np.radians(df_combined['AngleY_deg']))
df_combined['y'] = df_combined['Displacement'] * np.sin(np.radians(df_combined['AngleX_deg']))
df_combined['z'] = df_combined['Displacement'] * np.cos(np.radians(df_combined['AngleX_deg'])) * np.cos(np.radians(df_combined['AngleY_deg']))

# 将日期时间转换为数值（以秒为单位）
df_combined['time_numeric'] = (df_combined.index - df_combined.index[0]).total_seconds()

# 创建 3D 轨迹图
fig = go.Figure(data=[go.Scatter3d(
    x=df_combined['x'],
    y=df_combined['y'],
    z=df_combined['z'],
    mode='lines+markers',
    marker=dict(
        size=2,
        color=df_combined['time_numeric'],  # 使用数值时间
        colorscale='Viridis',
        opacity=0.8
    ),
    line=dict(
        color='darkblue',
        width=2
    )
)])

# 设置布局
fig.update_layout(
    title='Interactive 3D Trajectory',
    scene=dict(
        xaxis_title='X Position',
        yaxis_title='Y Position',
        zaxis_title='Z Position',
        aspectmode='data'  # 这将保持 x, y, z 轴的比例一致
    ),
    width=900,
    height=700,
)

# 显示图形
fig.show()

# 如果你想保存为 HTML 文件以便在浏览器中打开
# fig.write_html("3d_trajectory.html")

# 打印一些基本统计信息
print("轨迹统计信息:")
print(f"总位移: {df_combined['Displacement'].iloc[-1] - df_combined['Displacement'].iloc[0]:.4f}")
print(f"X 方向总变化: {df_combined['x'].max() - df_combined['x'].min():.4f}")
print(f"Y 方向总变化: {df_combined['y'].max() - df_combined['y'].min():.4f}")
print(f"Z 方向总变化: {df_combined['z'].max() - df_combined['z'].min():.4f}")
print(f"总时长: {df_combined.index[-1] - df_combined.index[0]}")
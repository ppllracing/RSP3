import pandas as pd
import matplotlib.pyplot as plt

# 读取文件并处理数据
file_path = '/home/jiao/AutomatedValetParking/solution/Solution_test.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, sep="\t")  # 读取带有制表符分隔的文件

# 提取 x 和 y 列
x = data['x']
y = data['y']

# 绘制轨迹图
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', markersize=3)
plt.title('Trajectory Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

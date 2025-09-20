import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 原始数据
x = np.arange(-1, 1.5, 0.5)
y = np.arange(-1, 1.5, 0.5)
x_grid, y_grid = np.meshgrid(x, y)
z = 1 - x_grid**2 - y_grid**2

# 新的插值点
new_x = np.arange(-1, 1.01, 0.1)
new_y = np.arange(-1, 1.01, 0.1)
new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)

# 二维插值，三次样条
points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
values = z.ravel()
p = griddata(points, values, (new_x_grid, new_y_grid), method='cubic')

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(new_x_grid, new_y_grid, p, cmap='viridis')
plt.show()

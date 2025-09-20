import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from scipy.interpolate import PchipInterpolator, CubicSpline

def get_chinese_font():
    """
    返回指定路径的微软雅黑字体 FontProperties 实例。
    """
    path = '/home/xhwang/msyh.ttc'
    if os.path.exists(path):
        return FontProperties(fname=path)
    else:
        # 字体文件不存在则回退到 matplotlib 默认字体（可能无法显示中文）
        return FontProperties()

plt.rcParams['axes.unicode_minus'] = False
chinese_font = get_chinese_font()

# 定义样本点
x = np.arange(1, 11)            # 1,2,...,10
y = np.log(x)                   # ln(x)

# 绘制样本点
plt.plot(x, y, 'o', label='样本点')

# 细分网格
new_x = np.arange(0.01, 10.01, 0.1)

# 三次埃尔米特插值（PCHIP）
pchip_interp = PchipInterpolator(x, y)
pchip_y = pchip_interp(new_x)
plt.plot(new_x, pchip_y, '-', label='三次埃尔米特插值')

# 三次样条插值
spline_interp = CubicSpline(x, y)
spline_y = spline_interp(new_x)
plt.plot(new_x, spline_y, 'b-', label='三次样条插值')

# 图形美化与显示
plt.legend(loc='lower right', prop=chinese_font)
plt.xlabel('x', fontproperties=chinese_font)
plt.ylabel('y', fontproperties=chinese_font)
plt.title('插值比较: ln(x)', fontproperties=chinese_font)
plt.grid(True)
plt.show()

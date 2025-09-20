import numpy as np
from tqdm import tqdm

np.random.seed(42)  # 固定随机数种子，保证结果可复现

def fun(x):
    # x: array-like, shape (2,)
    return x[0] * np.sin(np.pi * x[0]) + x[1] * np.cos(4 * np.pi * x[1])

# 参数初始化
narvs = 2  # 变量个数
T0 = 1000  # 初始温度
T_min = 1e-3  # 最低温度，防止无意义迭代
maxgen = 1000  # 最大迭代次数
Lk = 300  # 每个温度下的迭代次数
alfa = 0.95  # 温度衰减系数
x_lb = np.array([-3, 4])  # x的下界
x_ub = np.array([3, 5])   # x的上界

# 随机生成一个初始解
cur_x = x_lb + (x_ub - x_lb) * np.random.rand(narvs)
cur_y = fun(cur_x)

# 保存最优解
best_y = cur_y
best_x = cur_x.copy()

T = T0
iter_count = 0
with tqdm(total=maxgen, desc="SA迭代进度") as pbar:
    while T > T_min and iter_count < maxgen:
        for _ in range(Lk):
            # 生成新解：在当前解附近以温度为标准差的高斯扰动
            x_new = cur_x + np.random.randn(narvs) * (T / T0) * (x_ub - x_lb) * 0.1

            # 边界处理，直接clip
            x_new = np.clip(x_new, x_lb, x_ub)

            y_new = fun(x_new)
            delta = y_new - cur_y

            if delta > 0:
                cur_x = x_new
                cur_y = y_new
            else:
                p = np.exp(delta / T)
                if np.random.rand() < p:
                    cur_x = x_new
                    cur_y = y_new

            if cur_y > best_y:
                best_y = cur_y
                best_x = cur_x.copy()
        T *= alfa  # 温度衰减
        iter_count += 1
        pbar.update(1)

print("取最大值时的根是：", best_x)
print("此时对应的最大值是：", best_y)

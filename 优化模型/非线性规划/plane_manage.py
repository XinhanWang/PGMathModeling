import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

def set_chinese_font(font_path='/home/xhwang/msyh.ttc'):
    try:
        # 将字体文件加入 font manager（matplotlib >=3.2 支持）
        font_manager.fontManager.addfont(font_path)
        # 设置为无衬线字体族并指定字体名称（微软雅黑）
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Microsoft Yahei', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        # 若加载失败，尽量关闭 unicode_minus，继续运行
        matplotlib.rcParams['axes.unicode_minus'] = False

# 在模块导入时立即设置中文字体
set_chinese_font()

def optimize_step(positions, headings, v=8.0, d_min=12.0, max_delta=np.pi/6):
    """
    计算单步最优转向增量，使得更新后各飞机满足最小间隔约束并尽量少调整。
    positions: (n,2)
    headings: (n,)
    返回: delta_theta (n,)
    """
    n = len(headings)
    x0 = np.zeros(n)

    # 目标：最小化 ∑ delta^2
    def objective(d):
        return np.sum(d**2)

    # 计算更新后位置
    def updated_positions(d):
        new_head = headings + d
        disp = np.vstack((np.cos(new_head), np.sin(new_head))).T * v
        return positions + disp

    # 构造两两间隔不等式约束：dist_ij^2 - d_min^2 >= 0
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_fun(a, b):
                return lambda d, a=a, b=b: (
                    np.sum((updated_positions(d)[a] - updated_positions(d)[b])**2) - d_min**2
                )
            cons.append({'type': 'ineq', 'fun': make_fun(i, j)})

    bounds = [(-max_delta, max_delta)] * n

    res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP',
                   options={'ftol': 1e-6, 'maxiter': 200, 'disp': False})
    if not res.success:
        # 若失败，尝试放宽步长或直接返回零调整（可按需改进）
        # print("Warning: optimization failed:", res.message)
        return np.zeros(n)
    return res.x

def simulate(steps=20, v=None, speed=None, dt_minutes=60.0, d_min=12.0):
    """
    simulate 仿真函数：
    - 可以直接传入 v（每步位移，单位与位置单位一致，保持向后兼容）
    - 或者传入 speed（速度，单位：距离/小时）和 dt_minutes（每步的分钟数），
      此时每步位移 v = speed * (dt_minutes / 60)
    默认 dt_minutes=60 表示每步为 1 小时，与历史用法一致。
    """
    # 兼容：如果未给出每步位移 v，则用 speed 和 dt_minutes 计算
    if v is None:
        if speed is None:
            raise ValueError("请提供 v（每步位移）或 speed（速度，距离/小时）之一")
        v = speed * (dt_minutes / 60.0)

    # 初始数据 (x, y, 角度(度))
    raw = np.array([
        [150, 140, 243],
        [85,  85,  236],
        [150, 155, 220.5],
        [145, 50,  159],
        [130, 150, 230],
        [0,   0,   52]
    ], dtype=float)
    positions = raw[:, :2].copy()
    headings = np.deg2rad(raw[:, 2])
    traj = [positions.copy()]

    for _ in range(steps):
        delta = optimize_step(positions, headings, v=v, d_min=d_min)
        headings += delta
        positions += np.vstack((np.cos(headings), np.sin(headings))).T * v
        traj.append(positions.copy())

    return np.array(traj), headings

def plot_trajectory(traj):
    n_steps, n, _ = traj.shape
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(6, 6))
    for i in range(n):
        path = traj[:, i, :]
        plt.plot(path[:, 0], path[:, 1], '-', color=colors[i % len(colors)], lw=1)
        plt.plot(path[0, 0], path[0, 1], 'o', color=colors[i % len(colors)], ms=5)
        plt.plot(path[-1, 0], path[-1, 1], 's', color=colors[i % len(colors)], ms=5)
        plt.text(path[-1, 0] + 1, path[-1, 1] + 1, f"机{i+1}", fontsize=8)
    plt.title("飞机轨迹(非线性规划航向调整)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # 示例：以速度 800（单位/小时），每步为 1 分钟进行仿真（步长为 1 分钟）
    traj, final_headings = simulate(steps=10, speed=800.0, dt_minutes=1.0, d_min=8.0)
    print("最终朝向(度)：", np.rad2deg(final_headings))
    plot_trajectory(traj)

if __name__ == "__main__":
    main()

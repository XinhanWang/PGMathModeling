import numpy as np
import matplotlib.pyplot as plt
# 新增：中文字体支持
from matplotlib.font_manager import FontProperties

def get_chinese_font():
    """
    返回指定路径的微软雅黑字体 FontProperties 实例。
    """
    return FontProperties(fname='/home/xhwang/msyh.ttc')

plt.rcParams['axes.unicode_minus'] = False
chinese_font = get_chinese_font()

# 目标函数，支持标量和 numpy 数组
def f(x):
    x = np.asarray(x)
    return x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x)

def main():
    # 参数（与原 MATLAB 代码对应）
    N = 20         # 粒子数
    d = 1          # 维度
    ger = 100      # 最大迭代次数
    limit = np.array([0.0, 50.0])
    vlimit = np.array([-10.0, 10.0])
    w = 0.8
    c1 = 0.5
    c2 = 0.5

    # 初始化
    rng = np.random.default_rng()
    x = limit[0] + (limit[1] - limit[0]) * rng.random((N, d))
    v = rng.random((N, d))
    xm = x.copy()                        # 个体历史最佳位置
    fxm = np.full((N,), np.inf)          # 个体历史最佳适应度
    fym = np.inf                         # 群体历史最佳适应度
    ym = np.zeros(d)                     # 群体历史最佳位置

    # 绘图准备
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x0 = np.linspace(limit[0], limit[1], 1000)
    axes[0].plot(x0, f(x0), 'b-')  # 函数曲线
    axes[0].set_title('状态位置变化', fontproperties=chinese_font)
    axes[1].set_title('最优适应度进化过程', fontproperties=chinese_font)

    record = np.zeros(ger)

    for it in range(ger):
        fx = f(x.reshape(-1)) if d == 1 else f(x)  # 当前适应度
        fx = np.asarray(fx).reshape(N,)            # 保证形状正确

        # 更新个体历史最优
        better = fx < fxm
        if np.any(better):
            fxm[better] = fx[better]
            xm[better, :] = x[better, :]

        # 更新群体历史最优
        min_idx = np.argmin(fxm)
        if fxm[min_idx] < fym:
            fym = fxm[min_idx]
            ym = xm[min_idx, :].copy()

        # 速度与位置更新（随机项按标量处理以模拟 MATLAB rand 行为）
        r1 = rng.random()
        r2 = rng.random()
        v = v * w + c1 * r1 * (xm - x) + c2 * r2 * (np.tile(ym, (N, 1)) - x)

        # 速度边界
        v = np.clip(v, vlimit[0], vlimit[1])

        # 位置更新与边界处理
        x = x + v
        x = np.clip(x, limit[0], limit[1])

        record[it] = fym

        # 绘图更新
        axes[0].cla()
        axes[0].plot(x0, f(x0), 'b-')
        axes[0].plot(x.flatten(), f(x.flatten()), 'ro')
        axes[0].set_title('状态位置变化', fontproperties=chinese_font)

        axes[1].cla()
        axes[1].plot(record[:it+1])
        axes[1].set_title('最优适应度进化过程', fontproperties=chinese_font)

        plt.pause(0.01)

    # 最终结果显示
    plt.ioff()
    fig2, ax2 = plt.subplots()
    ax2.plot(x0, f(x0), 'b-')
    ax2.plot(x.flatten(), f(x.flatten()), 'ro')
    ax2.set_title('最终状态位置', fontproperties=chinese_font)
    plt.show()

    print('最小值：', fym)
    print('变量取值：', ym)

if __name__ == '__main__':
    main()

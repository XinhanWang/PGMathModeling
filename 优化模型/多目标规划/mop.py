from typing import List, Tuple
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
try:
    import geatpy as ga
except Exception as e:
    ga = None  # 若未安装 geatpy，后续 ga_moo 会抛出提示

# ----------------- 基础数据 -----------------
# 目标:
#   max Z1 = 9x1 + 10x2 + 14x3
#   min Z2 = 4x1 + 5x2 + 8x3
c1 = [9, 10, 14]
c2 = [4, 5, 8]

A = [
    [2, 8, 6],
    [4, 4, 6],
    [7, 6, 8],
]
b = [40, 50, 80]
n = 3

# ----------------- 通用建模工具 (改为使用 scipy.milp) -----------------
def solve_milp_int(c_vec, maximize=False, extra_A=None, extra_b=None):
    """
    使用 scipy.optimize.milp 求解整数规划：
    min c_vec^T x  subject to A_ub x <= b_ub, x >= 0, x 为整数
    若 maximize=True，则求解 max c_vec^T x (通过最小化 -c_vec)
    extra_A: list of rows 追加到 A_ub
    extra_b: list of rhs 追加到 b_ub
    """
    # 基本不等式 Ax <= b
    A_ub = [row[:] for row in A]
    b_ub = b[:]
    if extra_A is not None:
        for row in extra_A:
            A_ub.append(row[:])
    if extra_b is not None:
        b_ub.extend(extra_b)
    # 准备参数
    c_use = np.array(c_vec, dtype=float)
    if maximize:
        c_use = -c_use
    # 使用 Bounds 显式给出下界与上界数组，确保为实数且可广播到 c.shape
    bounds = Bounds(np.zeros(n, dtype=float), np.full(n, np.inf, dtype=float))
    # 所有变量为整数
    integrality = np.ones(n, dtype=int)
    # 构造 LinearConstraint
    constraints = None
    if A_ub:
        constraints = LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub))
    # 调用 milp（使用关键字参数以兼容不同 SciPy 版本）
    try:
        res = milp(c=c_use,
                   constraints=constraints,
                   bounds=bounds,
                   integrality=integrality,
                   options={"disp": False})
    except TypeError:
        # 某些 SciPy 版本对 options 或参数签名不同，尝试更简单的调用
        res = milp(c=c_use,
                   constraints=constraints,
                   bounds=bounds,
                   integrality=integrality)
    status = "Optimal" if getattr(res, "success", False) else getattr(res, "message", "Not Solved")
    x_vals = res.x.tolist() if hasattr(res, "x") else [0.0] * n
    obj = getattr(res, "fun", None)
    if maximize and obj is not None:
        obj = -obj
    # 将解四舍五入为整数以避免浮点微小误差
    try:
        x_vals = [int(round(v)) for v in x_vals]
    except Exception:
        pass
    return status, x_vals, obj

# ----------------- 线性加权法 (改为使用 milp) -----------------
def weighted_sum(w1: float, w2: float):
    # 统一转成最小化：f = w1*(-Z1) + w2*(Z2)
    c_vec = [w1 * (-c1[i]) + w2 * (c2[i]) for i in range(n)]
    status, xs, obj = solve_milp_int(c_vec, maximize=False)
    z1v = sum(c1[i] * xs[i] for i in range(n))
    z2v = sum(c2[i] * xs[i] for i in range(n))
    return {"status": status, "x": xs, "Z1": z1v, "Z2": z2v, "scalar_obj": obj, "w": (w1, w2)}

# ----------------- 优先级法 (词典序) (改为使用 milp) -----------------
def priority(primary: str = "Z1", eps: float = 1e-6):
    if primary == "Z1":
        # 第一步：最大化 Z1
        status1, xs1, Z1_star = solve_milp_int(c1, maximize=True)
        # 第二步：在 Z1 >= Z1_star - eps 下最小化 Z2
        # 将 Z1 >= Z1_star - eps 转化为 -c1 * x <= -(Z1_star - eps)
        extra_A = [[-ci for ci in c1]]
        extra_b = [-(Z1_star - eps)]
        status2, xs2, z2v = solve_milp_int(c2, maximize=False, extra_A=extra_A, extra_b=extra_b)
        z1v = sum(c1[i] * xs2[i] for i in range(n))
        return {"order": "Z1->Z2", "Z1*": Z1_star, "x1_stage": xs1, "final_x": xs2, "Z1": z1v, "Z2": z2v}
    else:
        # 先最小化 Z2，再最大化 Z1
        status1, xs1, Z2_star = solve_milp_int(c2, maximize=False)
        # 在 Z2 <= Z2_star + eps 下最大化 Z1  ->  c2 * x <= Z2_star + eps
        extra_A = [c2[:]]
        extra_b = [Z2_star + eps]
        status2, xs2, z1v = solve_milp_int(c1, maximize=True, extra_A=extra_A, extra_b=extra_b)
        z2v = sum(c2[i] * xs2[i] for i in range(n))
        return {"order": "Z2->Z1", "Z2*": Z2_star, "x1_stage": xs1, "final_x": xs2, "Z1": z1v, "Z2": z2v}

# ----------------- 理想点法 (改为 milp) -----------------
def single_optimize(maximize_Z1: bool):
    if maximize_Z1:
        status, xs, obj = solve_milp_int(c1, maximize=True)
    else:
        status, xs, obj = solve_milp_int(c2, maximize=False)
    return xs, obj

def ideal_point_method(weight_dist: Tuple[float, float] = (0.5, 0.5)):
    # 理想点
    _, Z1_star = single_optimize(True)
    _, Z2_star = single_optimize(False)
    # 反理想点(用于归一化)
    # min Z1
    _, xs_minZ1, _ = solve_milp_int(c1, maximize=False)
    Z1_min = sum(c1[i] * xs_minZ1[i] for i in range(n))
    # max Z2
    _, xs_maxZ2, _ = solve_milp_int(c2, maximize=True)
    Z2_max = sum(c2[i] * xs_maxZ2[i] for i in range(n))
    range1 = max(Z1_star - Z1_min, 1e-9)
    range2 = max(Z2_max - Z2_star, 1e-9)
    w1, w2 = weight_dist
    # 线性化 L1 目标的系数（去掉常数项）
    c_vec = [w1 * (-c1[i]) / range1 + w2 * (c2[i]) / range2 for i in range(n)]
    status, xs, obj = solve_milp_int(c_vec, maximize=False)
    z1v = sum(c1[i] * xs[i] for i in range(n))
    z2v = sum(c2[i] * xs[i] for i in range(n))
    return {
        "ideal": (Z1_star, Z2_star),
        "anti": (Z1_min, Z2_max),
        "x": xs,
        "Z1": z1v,
        "Z2": z2v,
        "scaled_obj": obj
    }

# ----------------- 简化 Geatpy NSGA-II 遗传算法 -----------------
# 使用 geatpy 模板（需要已安装 geatpy）
def ga_moo(pop_size=50, generations=120):
    """
    使用 geatpy-2.7.0 实现的 NSGA-II 求解本整数多目标规划：
        max Z1 = 9x1 + 10x2 + 14x3
        min Z2 = 4x1 + 5x2 + 8x3
        s.t.  A x <= b, x >= 0, x 整数
    返回：帕累托前沿解列表，每个元素包含 {x, Z1, Z2}
    """
    if ga is None:
        raise ImportError("未安装 geatpy，请先: pip install geatpy==2.7.0")
    ver = getattr(ga, "__version__", "")
    if ver and ver < "2.7.0":
        raise RuntimeError(f"检测到 geatpy 版本为 {ver}，请安装 2.7.0 以运行该实现。")

    # 估计每个变量的合理上界（基于约束的保守取整，避免搜索空间过大）
    ub_list = []
    for i in range(n):
        feas_bounds = []
        for row, bi in zip(A, b):
            coef = row[i]
            if coef > 0:
                feas_bounds.append(bi // coef)
        if feas_bounds:
            ub_list.append(int(max(0, min(feas_bounds))))
        else:
            ub_list.append(int(max(b)))
    lb_list = [0] * n

    class MOPProblem(ga.Problem):
        def __init__(self):
            name = "MOP_NSGA2_v270"
            M = 2
            maxormins = [-1, 1]          # Z1 最大化, Z2 最小化
            Dim = n
            varTypes = [1] * n           # 1 表示整数变量
            lb = lb_list
            ub = ub_list
            lbin = [1] * n               # 下界可取
            ubin = [1] * n               # 上界可取
            super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        def aimFunc(self, pop):
            X = pop.Phen.astype(float)                         # (N, n)
            coeff1 = np.array(c1, dtype=float)
            coeff2 = np.array(c2, dtype=float)
            Z1 = X.dot(coeff1)
            Z2 = X.dot(coeff2)
            pop.ObjV = np.vstack([Z1, Z2]).T                  # (N, 2)
            # 约束：A x <= b  ->  G = A x - b <= 0
            A_mat = np.array(A, dtype=float)                  # (m, n)
            b_vec = np.array(b, dtype=float)                  # (m,)
            G = X.dot(A_mat.T) - b_vec                         # (N, m)
            pop.CV = G                                         # geatpy: <=0 可行

    # 创建问题与种群
    problem = MOPProblem()
    Encoding = "RI"  # 整数/实数混合编码
    # geatpy 2.7.0 中 Problem 会暴露 ranges & borders
    try:
        Field = ga.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    except Exception as e:
        raise RuntimeError(f"创建 Field 失败，请确认 geatpy 版本: {e}")
    population = ga.Population(Encoding, Field, pop_size)

    # 构建 NSGA-II 算法模板
    algorithm = ga.moea_NSGA2_templet(problem, population)
    algorithm.MAXGEN = generations
    algorithm.drawing = 1          # 绘图
    # 可选参数（收敛控制）
    algorithm.trappedValue = 1e-6
    algorithm.maxTrappedCount = 20

    NDSet = algorithm.run()[0]        # NDSet 为非支配解集
    if NDSet is None or getattr(NDSet, "Phen", None) is None:
        return []

    phen = NDSet.Phen
    objv = NDSet.ObjV
    results = []
    for i in range(phen.shape[0]):
        xs_raw = phen[i].tolist()
        xs = [int(round(v)) for v in xs_raw]
        z1 = float(objv[i, 0])
        z2 = float(objv[i, 1])
        # 由于整数修正，重新精确计算目标
        z1 = sum(c1[j] * xs[j] for j in range(n))
        z2 = sum(c2[j] * xs[j] for j in range(n))
        # 过滤不可行（保险：若数值扰动导致约束失真）
        feasible = True
        for row, bi in zip(A, b):
            if sum(row[j] * xs[j] for j in range(n)) > bi:
                feasible = False
                break
        if feasible:
            results.append({"x": xs, "Z1": z1, "Z2": z2})
    # 去重（可能不同个体映射到同一整数解）
    uniq = {}
    for r in results:
        uniq[tuple(r["x"])] = r
    return list(uniq.values())

# ----------------- 对线性加权法进行扫描并画图 -----------------
def get_chinese_font():
    """
    返回指定路径的微软雅黑字体 FontProperties 实例。
    若无法加载（文件不存在或 matplotlib 未安装），返回 None 以作回退。
    """
    try:
        from matplotlib.font_manager import FontProperties
        # 用户提供的字体路径
        fp_path = '/home/xhwang/msyh.ttc'
        try:
            fp = FontProperties(fname=fp_path)
            return fp
        except Exception:
            # 字体文件无法加载
            return None
    except Exception:
        # matplotlib 未安装或无法导入 FontProperties
        return None

def scan_and_plot_weighted_sum(num_points: int = 101, save_path: str = None, show: bool = True):
    """
    在 w1 从 0 到 1 上等间距采样 num_points 个点，计算加权和法得到的 (Z1,Z2)。
    - 返回列表，每项为 dict: {"w": (w1,w2), "x": x, "Z1": z1, "Z2": z2}
    - 若安装了 matplotlib，会绘制以 w1 为横坐标、Z1 和 Z2 为纵坐标的折线图，点颜色表示 w1。
    - save_path: 若提供则保存图片到该路径。
    """
    ws = np.linspace(0.0, 1.0, num_points)
    results = []
    for w1 in ws:
        w2 = 1.0 - w1
        res = weighted_sum(w1, w2)
        results.append({"w": (w1, w2), "x": res["x"], "Z1": res["Z1"], "Z2": res["Z2"]})

    # 尝试绘图（若 matplotlib 可用）
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib 未安装，已完成扫描但无法画图。可通过 `pip install matplotlib` 安装。")
        return results

    # 准备绘图数据：横坐标为 w1，纵坐标为 Z1 与 Z2
    w1s = [r["w"][0] for r in results]
    Z1s = [r["Z1"] for r in results]
    Z2s = [r["Z2"] for r in results]

    plt.figure(figsize=(8,5))
    # 绘制 Z1 与 Z2 随 w1 变化的折线
    plt.plot(w1s, Z1s, label="Z1", color="tab:blue", marker="o", markersize=4, linewidth=1.5)
    plt.plot(w1s, Z2s, label="Z2", color="tab:orange", marker="s", markersize=4, linewidth=1.5)
    # 标注极端点
    idx_maxZ1 = int(np.argmax(Z1s))
    idx_minZ2 = int(np.argmin(Z2s))
    plt.scatter([w1s[idx_maxZ1]], [Z1s[idx_maxZ1]], c="red", marker="*", s=100, label="max Z1")
    plt.scatter([w1s[idx_minZ2]], [Z2s[idx_minZ2]], c="cyan", marker="D", s=80, label="min Z2")

    # 尝试获取中文字体（若可用则在中文标签/标题中使用）
    font = get_chinese_font()
    if font is not None:
        plt.xlabel("w1", fontproperties=font)
        plt.ylabel("目标值", fontproperties=font)
        plt.title("权重 vs 目标值：Z1 与 Z2 随 w1 的变化", fontproperties=font)
        plt.legend(prop=font)
    else:
        plt.xlabel("w1")
        plt.ylabel("objective value")
        plt.title("Weight vs Objectives: Z1 and Z2 vs w1")
        plt.legend()

    plt.grid(True)
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"已保存图片到: {save_path}")
        except Exception as e:
            print("保存图片失败:", e)
    if show:
        plt.show()
    plt.close()
    return results

# ----------------- 演示主程序 -----------------
def main():
    print("=== 线性加权法 示例 (w1 + w2 = 1) ===")
    for w1 in [i / 10 for i in range(11)]:
        w2 = 1 - w1
        res = weighted_sum(w1, w2)
        print(f"w=({w1:.1f},{w2:.1f}) -> Z1={res['Z1']:.3f}, Z2={res['Z2']:.3f}, x={['%.3f'%v for v in res['x']]}")
    # 新增：对加权和法进行扫描并画图
    try:
        scan_and_plot_weighted_sum(num_points=101, save_path=None, show=True)
    except Exception as e:
        print("扫描并绘图时出错:", e)

    print("\n=== 优先级法 Z1->Z2 ===")
    print(priority("Z1"))
    print("\n=== 优先级法 Z2->Z1 ===")
    print(priority("Z2"))
    print("\n=== 理想点法 ===")
    print(ideal_point_method())
    print("\n=== 遗传算法 (近似帕累托前沿) ===")
    try:
        pareto = ga_moo()
        for p in pareto[:10]:
            print(f"x={['%.3f'%v for v in p['x']]} Z1={p['Z1']:.3f} Z2={p['Z2']:.3f}")
        print(f"Pareto 点数量: {len(pareto)}")
    except ImportError as e:
        print(str(e))
    except Exception as e:
        print("运行 geatpy 遗传算法时出错:", e)

if __name__ == "__main__":
    main()

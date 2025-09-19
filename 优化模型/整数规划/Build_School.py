import random
import argparse

def monte_carlo_build_school(n=10000, seed=None):
    """
    用蒙特卡洛方法搜索满足约束的最小建校方案。
    返回 (res_min, res_x)：
      - res_min: 最小选点数（若未找到则为 None）
      - res_x: 对应的 6 元 0/1 列表（若未找到则为 None）
    约束与原 MATLAB 代码一致（x 索引从 0 到 5 对应 x1..x6）：
      x1 + x2 + x3 >= 1
      x4 + x6 >= 1
      x3 + x5 >= 1
      x2 + x4 >= 1
      x1 >= 1
      x2 + x4 + x6 >= 1
    """
    if seed is not None:
        random.seed(seed)
    res_min = None
    res_x = None
    for _ in range(n):
        x = [random.randint(0, 1) for _ in range(6)]
        if ((x[0] + x[1] + x[2] >= 1) and
            (x[3] + x[5] >= 1) and
            (x[2] + x[4] >= 1) and
            (x[1] + x[3] >= 1) and
            (x[0] >= 1) and
            (x[1] + x[3] + x[5] >= 1)):
            sum_x = sum(x)
            if (res_min is None) or (sum_x < res_min):
                res_min = sum_x
                res_x = x.copy()
    return res_min, res_x

def solve_build_school_milp():
    """
    使用 scipy.optimize.milp 精确求解最小选点问题。
    返回 (res_min, res_x)：最小选点数及对应的 6 元 0/1 列表（若无解则返回 (None, None)）。
    约束与原 MATLAB 代码一致（索引 0..5 对应 x1..x6），且 x1 必须为 1。
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np

    # 目标函数：min x1 + x2 + ... + x6
    c = np.ones(6)

    # 约束矩阵与右端项
    A = np.array([
        [1, 1, 1, 0, 0, 0],   # x1 + x2 + x3 >= 1
        [0, 0, 0, 1, 0, 1],   # x4 + x6 >= 1
        [0, 0, 1, 0, 1, 0],   # x3 + x5 >= 1
        [0, 1, 0, 1, 0, 0],   # x2 + x4 >= 1
        [1, 0, 0, 0, 0, 0],   # x1 >= 1
        [0, 1, 0, 1, 0, 1],   # x2 + x4 + x6 >= 1
    ])
    lb = np.ones(6)  # 所有约束右端项均为 >= 1
    ub = np.full(6, np.inf)  # 无上界

    constraints = LinearConstraint(A, lb, ub)

    # 变量界限：0 <= xi <= 1
    bounds = Bounds(0, 1)

    # 整数变量
    integrality = np.ones(6, dtype=int)

    # 求解
    res = milp(c, constraints=[constraints], bounds=bounds, integrality=integrality)

    if res.success and res.x is not None:
        x_sol = [int(round(v)) for v in res.x]
        res_min = sum(x_sol)
        return res_min, x_sol
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser(description="求解建校问题（Python 版，支持 monte|milp）")
    parser.add_argument("-n", type=int, default=10000, help="蒙特卡洛模拟次数，默认 10000（仅在 solver=monte 时生效）")
    parser.add_argument("--seed", type=int, default=1, help="随机数种子（可选）")
    parser.add_argument("--solver", choices=["monte", "milp"], default="milp", help="选择求解器：monte 或 milp，默认 milp")
    args = parser.parse_args()

    if args.solver == "monte":
        res_min, res_x = monte_carlo_build_school(n=args.n, seed=args.seed)
        if res_min is None:
            print("在给定的模拟次数内未找到满足约束的解。")
        else:
            print(f"找到的最小选点数: {res_min}")
            print(f"对应的选点向量 (x1..x6): {res_x}")
    else:
        res_min, res_x = solve_build_school_milp()
        if res_min is None:
            print("MILP 未找到可行解或求解失败。")
        else:
            print(f"MILP 求得最小选点数: {res_min}")
            print(f"对应的选点向量 (x1..x6): {res_x}")

if __name__ == "__main__":
    main()

import random

def monte_carlo_zhangmazi(n=10000, seed=None):
    """
    用蒙特卡洛方法解决张麻子问题。
    返回 (max_y, res_x1, res_x2)
    """
    if seed is not None:
        random.seed(seed)
    max_y = -float('inf')
    res_x1 = None
    res_x2 = None
    for _ in range(n):
        x1 = random.randint(1,6)
        x2 = random.randint(1,6)
        if x1 + x2 <= 6 and 240*x1 + 120*x2 <= 1200:
            y = 40*x1 + 30*x2
            if y > max_y:
                max_y = y
                res_x1 = x1
                res_x2 = x2
    if res_x1 is not None:
        return max_y, res_x1, res_x2
    else:
        return None, None, None

def milp_zhangmazi():
    """
    用 MILP 精确求解张麻子问题。
    返回 (max_y, x1, x2)
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np

    # 决策变量 x1, x2
    # x1: 1~5, x2: 1~6, 都是整数
    c = [-40, -30]  # 最大化 40*x1 + 30*x2，milp默认最小化，所以取负号

    # 约束
    # x1 + x2 <= 6
    # 240*x1 + 120*x2 <= 1200
    # x1 >= 1, x1 <= 6
    # x2 >= 1, x2 <= 6
    A = [
        [1, 1],
        [240, 120]
    ]
    ub = [6, 1200]
    constraints = [LinearConstraint(A, [-np.inf, -np.inf], ub)]
    bounds = Bounds([1, 1], [6, 6])
    integrality = np.ones(2, dtype=int)

    res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    if res.success and res.x is not None:
        x1, x2 = np.round(res.x).astype(int)
        max_y = 40*x1 + 30*x2
        return max_y, x1, x2
    else:
        return None, None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="张麻子问题求解（Python版，支持 monte|milp）")
    parser.add_argument("--solver", choices=["monte", "milp"], default="milp", help="选择求解器：monte 或 milp，默认 milp")
    parser.add_argument("-n", type=int, default=10000, help="蒙特卡洛模拟次数（仅在 solver=monte 时生效）")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（可选）")
    args = parser.parse_args()

    if args.solver == "monte":
        max_y, x1, x2 = monte_carlo_zhangmazi(n=args.n, seed=args.seed)
        if x1 is not None:
            print(f"最大目标值: {max_y}")
            print(f"最优解: x1={x1}, x2={x2}")
        else:
            print("未找到满足约束的解。")
    else:
        max_y, x1, x2 = milp_zhangmazi()
        if x1 is not None:
            print(f"MILP 最大目标值: {max_y}")
            print(f"最优解: x1={x1}, x2={x2}")
        else:
            print("MILP 未找到可行解或求解失败。")

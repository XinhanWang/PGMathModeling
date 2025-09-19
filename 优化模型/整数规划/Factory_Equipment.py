import random

def monte_carlo_factory_equipment(n=10000, seed=None):
    """
    用蒙特卡洛方法解决工厂设备分配问题。
    返回 (res, res_x):
      - res: 最大总效益
      - res_x: 最优分配方案（长度为6的列表，每个元素为1~4，表示设备分配给哪个企业）
    """
    if seed is not None:
        random.seed(seed)
    C = [
        [4,2,3,4],
        [6,4,5,5],
        [7,6,7,6],
        [7,8,8,6],
        [7,9,8,6],
        [7,10,8,6]
    ]
    res = 0
    res_x = None
    for _ in range(n):
        x = [random.randint(1,4) for _ in range(6)]
        # 检查每个企业至少分配一台设备
        if all(j in x for j in range(1,5)):
            total = sum(C[i][x[i]-1] for i in range(6))
            if total > res:
                res = total
                res_x = x.copy()
    return res, res_x

def milp_factory_equipment():
    """
    使用 MILP 精确求解工厂设备分配问题。
    返回 (res, res_x):
      - res: 最大总效益
      - res_x: 最优分配方案（长度为6的列表，每个元素为1~4，表示设备分配给哪个企业）
    """
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    C = np.array([
        [4,2,3,4],
        [6,4,5,5],
        [7,6,7,6],
        [7,8,8,6],
        [7,9,8,6],
        [7,10,8,6]
    ])
    # 决策变量 x[i][j]: 第i台设备分配给第j个企业（i=0..5, j=0..3），共24个变量
    # x_flat: x[0][0], x[0][1], x[0][2], x[0][3], x[1][0], ..., x[5][3]
    c = -C.flatten()  # 最大化效益，milp默认最小化，所以取负号

    # 约束1：每台设备只能分配给一个企业
    A_eq = []
    b_eq = []
    for i in range(6):
        row = [0]*24
        for j in range(4):
            row[i*4 + j] = 1
        A_eq.append(row)
        b_eq.append(1)
    # 约束2：每个企业至少分配一台设备
    A_ub = []
    b_ub = []
    for j in range(4):
        row = [0]*24
        for i in range(6):
            row[i*4 + j] = -1  # -sum x[i][j] <= -1  <=> sum x[i][j] >= 1
        A_ub.append(row)
        b_ub.append(-1)
    # 变量界限
    bounds = Bounds([0]*24, [1]*24)
    integrality = np.ones(24, dtype=int)

    # 合并约束
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    constraints = []
    if len(A_eq) > 0:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    if len(A_ub) > 0:
        constraints.append(LinearConstraint(A_ub, [-np.inf]*len(b_ub), b_ub))

    res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
    if res.success and res.x is not None:
        x_sol = np.round(res.x).astype(int)
        # 还原分配方案
        assign = []
        for i in range(6):
            for j in range(4):
                if x_sol[i*4 + j] == 1:
                    assign.append(j+1)  # 企业编号1~4
                    break
        total = int(-res.fun)
        return total, assign
    else:
        return None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="工厂设备分配问题求解（Python版，支持 monte|milp）")
    parser.add_argument("--solver", choices=["monte", "milp"], default="milp", help="选择求解器：monte 或 milp，默认 milp")
    parser.add_argument("-n", type=int, default=10000, help="蒙特卡洛模拟次数（仅在 solver=monte 时生效）")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（可选）")
    args = parser.parse_args()

    if args.solver == "monte":
        res, res_x = monte_carlo_factory_equipment(n=args.n, seed=args.seed)
        if res_x is not None:
            print(f"最大总效益: {res}")
            print(f"最优分配方案（每台设备分配给企业编号1~4）: {res_x}")
        else:
            print("未找到满足约束的分配方案。")
    else:
        res, res_x = milp_factory_equipment()
        if res_x is not None:
            print(f"MILP 最大总效益: {res}")
            print(f"最优分配方案（每台设备分配给企业编号1~4）: {res_x}")
        else:
            print("MILP 未找到可行解或求解失败。")

import numpy as np
from scipy.optimize import minimize

# 常量参数
p1, p2 = 3.0, 1.0
k1, k2 = 1.0, 3.0
a = np.array([1.37, 9.45, 4.43, 6.66, 3.14, 15.92])
b = np.array([10.21, 9.45, 8.88, 5.00, 16.44, 18.00])
x1, y1 = 4.0, 1.0
x2, y2 = 8.0, 9.0
beq = np.array([5, 9, 4, 8, 14, 10], dtype=float)

# 预计算距离
D1 = np.sqrt((a - x1) ** 2 + (b - y1) ** 2)
D2 = np.sqrt((a - x2) ** 2 + (b - y2) ** 2)

def objective(x):
    # x 前六与后六
    xA = x[:6]
    xB = x[6:]
    term_linear = p1 * np.sum(xA) + p2 * np.sum(xB)
    # 避免除零
    # fracA = xA / (xA + 1e-6)
    # fracB = xB / (xB + 1e-6)
    # 使用符号函数
    fracA = np.sign(xA)
    fracB = np.sign(xB)
    term_dist = np.sum(k1 * D1 * fracA) + np.sum(k2 * D2 * fracB)
    return term_linear + 2*term_dist

# 等式约束: x[i] + x[i+6] = beq[i]
def eq_i(i):
    return lambda x, i=i: x[i] + x[i + 6] - beq[i]

# 不等式约束 (SLSQP 形式: fun(x) >= 0)
def ineq_sum_first(x):
    return 30.0 - np.sum(x[:6])

def ineq_sum_second(x):
    return 20.0 - np.sum(x[6:])

constraints = []
# 添加 6 个等式
for i in range(6):
    constraints.append({'type': 'eq', 'fun': eq_i(i)})
# 添加两个不等式
constraints.append({'type': 'ineq', 'fun': ineq_sum_first})
constraints.append({'type': 'ineq', 'fun': ineq_sum_second})

bounds = [(0.0, None)] * 12

# 可行性快速检测：
# sum(x[:6]) + sum(x[6:]) = sum(beq) = 51
# 但不等式要求 sum(x[:6]) <=30 且 sum(x[6:]) <=20 => 总和 <=50 矛盾 -> 不可行
total_beq = beq.sum()
if total_beq > 30 + 20:
    print("警告: 约束组不可行 (等式总和 {:.1f} > 不等式上界总和 50.0)。".format(total_beq))
    print("若需演示求解，可放宽不等式上界或调整 beq。示例: 将上界改为 (31, 21) 或修改 beq。")

# 构造一个“接近”可行的初值：先忽略不等式，平均分配
x0 = np.zeros(12)
# 这里给一个均分方案（不可行，但作为起点）
x0[:6] = beq / 2
x0[6:] = beq / 2

# 如需产生可行模型，可放宽上界 (示例):
# def ineq_sum_first(x): return 31.0 - np.sum(x[:6])
# def ineq_sum_second(x): return 21.0 - np.sum(x[6:])

res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
               options={'ftol': 1e-9, 'maxiter': 1000, 'disp': True})

print("\n=== 优化结果 (原始约束) ===")
print("成功标记:", res.success)
print("消息:", res.message)
print("目标函数值:", res.fun)
print("解向量:", res.x)

# 评估约束残差
eq_res = [eq_i(i)(res.x) for i in range(6)]
ineq_res = [ineq_sum_first(res.x), ineq_sum_second(res.x)]
print("等式残差:", eq_res)
print("不等式剩余(>=0表示满足):", ineq_res)

# 如果需要自动放宽并再次求解，可取消下段注释:
"""
if not res.success:
    print("\\n尝试放宽不等式上界以获得可行解...")
    def ineq_sum_first_relaxed(x): return 31.0 - np.sum(x[:6])
    def ineq_sum_second_relaxed(x): return 21.0 - np.sum(x[6:])
    constraints_relaxed = []
    for i in range(6):
        constraints_relaxed.append({'type': 'eq', 'fun': eq_i(i)})
    constraints_relaxed.append({'type': 'ineq', 'fun': ineq_sum_first_relaxed})
    constraints_relaxed.append({'type': 'ineq', 'fun': ineq_sum_second_relaxed})
    res2 = minimize(objective, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints_relaxed,
                    options={'ftol':1e-9,'maxiter':1000,'disp':True})
    print("放宽后成功:", res2.success, "目标:", res2.fun)
"""

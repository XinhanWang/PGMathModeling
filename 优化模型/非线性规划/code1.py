"""
scipy.optimize.minimize 用法简介：

minimize(fun, x0, args=(), method=None, bounds=None, constraints=(), ...)
- fun: 目标函数，形式为 fun(x)
- x0: 初始猜测值（数组）
- bounds: 变量范围 [(min, max), ...]
- constraints: 约束条件，字典列表，支持类型 'eq'（等式约束）和 'ineq'（不等式约束）
    例：{'type': 'eq', 'fun': lambda x: ...}
        {'type': 'ineq', 'fun': lambda x: ...}
- 返回值为优化结果对象 res，可通过 res.x 获取最优解，res.fun 获取最优目标函数值

详细文档见：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""

import numpy as np
from scipy.optimize import minimize

# 第一个例子
def f1(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + 8

def nonlcon1(x):
    # 非线性不等式约束
    c = [
        x[0] + x[1]**2 + x[2]**2 - 20  # <= 0
    ]
    # 非线性等式约束
    ceq = [
        -x[0] - x[1]**2 + 2  # == 0
    ]
    return c, ceq

x0 = [1, 1, 1]
A = np.array([[-1, 1, -1]])
b = np.array([0])

cons = [
    {'type': 'ineq', 'fun': lambda x: b[0] - np.dot(A[0], x)},  # -x1 + x2 - x3 <= 0
    {'type': 'ineq', 'fun': lambda x: -nonlcon1(x)[0][0]},      # x1 + x2^2 + x3^2 - 20 <= 0
    {'type': 'eq',   'fun': lambda x: nonlcon1(x)[1][0]},       # -x1 - x2^2 + 2 == 0
]

res1 = minimize(f1, x0, constraints=cons)
if res1.success:
    print("第一个例子结果：", res1.x, "目标函数值：", res1.fun)
else:
    print("第一个例子优化失败：", res1.message)

# 第二个例子
def f2(x):
    return -x[0]*x[1]*x[2]

def nonlcon2(x):
    c = [
        x[0] + x[1]**2 + x[2]**2 - 20,   # <= 0
        x[0]*x[1] - x[2] - 1             # <= 0
    ]
    ceq = [
        -x[0] - x[1]**2 + 2              # == 0
    ]
    return c, ceq

x0 = [1, 1, 0]
A = np.array([[-1, 1, -1]])
b = np.array([0])
lb = [0, -np.inf, -np.inf]
ub = [20, np.inf, np.inf]

bounds = [(lb[i], ub[i]) for i in range(3)]

cons2 = [
    {'type': 'ineq', 'fun': lambda x: b[0] - np.dot(A[0], x)},           # -x1 + x2 - x3 <= 0
    {'type': 'ineq', 'fun': lambda x: -nonlcon2(x)[0][0]},               # x1 + x2^2 + x3^2 - 20 <= 0
    {'type': 'ineq', 'fun': lambda x: -nonlcon2(x)[0][1]},               # x1*x2 - x3 - 1 <= 0
    {'type': 'eq',   'fun': lambda x: nonlcon2(x)[1][0]},                # -x1 - x2^2 + 2 == 0
]

res2 = minimize(f2, x0, bounds=bounds, constraints=cons2)
if res2.success:
    print("第二个例子结果：", res2.x, "目标函数值：", -res2.fun)
else:
    print("第二个例子优化失败：", res2.message)

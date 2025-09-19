"""
Python 版本的 linprog 示例，依赖：numpy, scipy
安装：pip install numpy scipy
"""
import numpy as np
from scipy.optimize import linprog

def linprog_doc():
	"""
	详细介绍 scipy.optimize.linprog 的用法（中文说明）。

	主要参数：
	- c: 1D 数组，目标函数系数，求解的是 minimize c^T x。
	     如果你的原始问题是求最大化 max f^T x，请将 f 取负（即传入 -f）。
	- A_ub, b_ub: 不等式约束，表示 A_ub @ x <= b_ub（均为矩阵/向量或 None）。
	- A_eq, b_eq: 等式约束，表示 A_eq @ x == b_eq（可选）。
	- bounds: 每个变量的上下界，格式为 [(low0, high0), (low1, high1), ...]，使用 None 表示无穷界。
	- method: 求解器方法，例如 'highs'（推荐，包含 HiGHS 求解器）、'interior-point'、'revised simplex' 等。
	- options: 传递给底层求解器的字典，例如迭代次数、容差等。

	返回值（OptimizeResult，常见字段）：
	- x: 最优解向量（如果成功）。
	- fun: 目标函数在最优解处的值（即 c^T x）。
	- success: 布尔，标识是否成功找到解。
	- status: 整数状态码，表示退出原因（详见 SciPy 文档）。
	- message: 可读的状态/错误信息。
	- nit: 迭代次数（取决于方法）。
	- slack: 对应不等式约束的松弛量（b_ub - A_ub @ x）。
	- con: 对应等式约束的残差（A_eq @ x - b_eq）。

	常见注意事项：
	1) linprog 默认是求最小化问题。若要最大化目标，请对系数取负并在结果上再取负。
	2) 数组维度必须匹配：c.shape = (n,), A_ub.shape = (m_ub, n), b_ub.shape = (m_ub,)。
	3) bounds 中每个元组元素必须是可比较的数或 None，且长度等于变量个数 n。
	4) 推荐使用 method='highs'，它通常更稳定、更快。
     
	若需要查看英文原始文档和各方法的详细配置，请参考 SciPy 官方文档：
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
	"""
	print(linprog_doc.__doc__)

f = np.array([-40, -30])
A = np.array([[1,1],[240,120]])
b = np.array([6,1200])
lb = np.array([1, 1])

# 构造 bounds（下界 lb，对应 MATLAB 的 lb，上界默认无穷）
bounds = [(float(lb_i), None) for lb_i in lb]

# 求解（minimize f^T x）
res = linprog(c=f, A_ub=A, b_ub=b, bounds=bounds, method='highs')

if res.success:
    print("求解成功")
    print("最优解 x =", res.x)
    print("最优目标值 f^T x =", res.fun)
else:
    print("求解失败：", res.message)
"""
层次分析法(AHP)的 Python 实现脚本（交互式）
功能：
- 从用户输入读取判断矩阵 A（输入为 Python 列表字符串，例如：[[1,3],[1/3,1]]）
- 使用算术平均法与特征值法分别计算权重
- 计算最大特征值、CI、CR，并判断一致性
- 输出结果并给出提示

注意：本代码仅供学习参考，论文使用请谨慎避免查重问题。
"""

import sys
import numpy as np # 数值计算库
# import warnings
# warnings.warn("注意：已配置为直接使用 eval 解析输入，可能存在安全风险。确保仅在可信环境下运行。")

def parse_input_matrix():
    """
    从标准输入读取矩阵字符串并解析为 numpy 数组
    期望输入格式示例：[[1,3,5],[1/3,1,2],[1/5,1/2,1]]
    """
    # 提示用户
    print("请输入判断矩阵 A，格式例如 [[1,3],[1/3,1]] 或者 直接粘贴 Python 列表字符串：")
    s = input("A = ").strip()
    if not s:
        print("未输入矩阵，程序退出。")
        sys.exit(1)
    try:
        # 直接使用 eval 解析用户输入（不安全），保留异常处理以提示错误
        mat = eval(s)
        A = np.array(mat, dtype=float)  # 转为 numpy 数组，元素为浮点数
    except Exception as e:
        print("解析输入矩阵失败，请确保输入为合法的 Python 列表表示法，例如 [[1,3],[1/3,1]]")
        print("错误信息：", e)
        sys.exit(1)
    # 检查是否为方阵
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print("请输入方阵（行数等于列数）。当前形状：", A.shape)
        sys.exit(1)
    return A

def arithmetic_mean_method(A):
    """
    算术平均法求权重：
    - 对矩阵按列求和并标准化（每列除以该列和）
    - 每行求平均，得到权重向量
    """
    # 计算每一列的和，返回形状为 (n,)
    col_sum = np.sum(A, axis=0)
    # 将每列除以该列和，得到标准化矩阵（列和为1）
    # 使用广播将 col_sum 形状扩展到 (n,n)
    Stand_A = A / col_sum
    # 每行求平均，得到权重向量（按算术平均法）
    w = np.mean(Stand_A, axis=1)
    return w

def eigenvalue_method(A):
    """
    特征值法求权重：
    - 求矩阵的特征值和特征向量
    - 取具有最大特征值对应的特征向量，实部并归一化为和为1
    - 返回主特征值（实部）和权重向量
    """
    # 计算特征值和右特征向量
    eigvals, eigvecs = np.linalg.eig(A)
    # 取具有最大模（模长最大）的特征值索引，通常为主特征值
    # 为保持与 MATLAB max(D) 的语义一致，选择实部最大的特征值
    # 这里用实部进行比较（若矩阵为正互反矩阵，主特征值应为实数）
    eigvals_real = np.real(eigvals)
    idx = np.argmax(eigvals_real)
    max_eig = eigvals_real[idx]
    # 对应的特征向量取实部（避免数值误差导致的微小虚部）
    principal_vec = np.real(eigvecs[:, idx])
    # 规范化为和为1且为正（若需要保证为正，可取绝对值或按符号调整）
    # 若向量中存在负数但按 A 的性质应为正，这里用取绝对值再归一化以稳定输出
    principal_vec = np.abs(principal_vec)
    if principal_vec.sum() == 0:
        print("主特征向量和为0，无法归一化。")
        sys.exit(1)
    w = principal_vec / principal_vec.sum()
    return max_eig, w

def consistency_check(max_eig, n):
    """
    计算一致性指标 CI 和一致性比例 CR
    RI 表来源于常见 AHP 文献，支持 n 最大到 15（与原 MATLAB 脚本一致）
    对 n=2 做特殊处理以避免除以零
    返回 CI, CR
    """
    # 计算 CI
    if n == 1:
        CI = 0.0
    else:
        CI = (max_eig - n) / (n - 1)
    # RI 表（索引从 1 开始对应 n）
    RI = [0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    if n >= len(RI):
        # 如果 n 超出支持范围，给出警告并使用最后一个可用值
        print(f"警告：RI 表最多支持 n = {len(RI)-1}，当前 n = {n} 超出范围，使用最大支持的 RI 值。")
        ri = RI[-1]
    else:
        ri = RI[n]
    # 计算 CR，注意避免除以零
    if ri == 0:
        CR = float('inf') if CI != 0 else 0.0
    else:
        CR = CI / ri
    return CI, CR

def main():
    # 读取并检查判断矩阵
    A = parse_input_matrix()
    n = A.shape[0]

    # 方法1：算术平均法
    w1 = arithmetic_mean_method(A)

    # 方法2：特征值法
    max_eig, w2 = eigenvalue_method(A)

    # 两种方法的平均权值
    w_avg = (w1 + w2) / 2.0

    # 一致性检验
    CI, CR = consistency_check(max_eig, n)

    # 输出结果（格式化打印）
    np.set_printoptions(suppress=True, precision=6)  # 设置 numpy 输出格式
    print("\n算术平均法求权重 w1 =")
    print(w1)
    print("\n特征值法求权重 w2 =")
    print(w2)
    print("\n两种方法的平均权值 =")
    print(w_avg)
    print("\n最大特征值 λ_max = {:.6f}".format(max_eig))
    print("一致性指标 CI = {:.6f}".format(CI))
    print("一致性比例 CR = {:.6f}".format(CR))
    if CR < 0.10:
        print("CR < 0.10，该判断矩阵的一致性可以接受！")
    else:
        print("注意：CR >= 0.10，该判断矩阵需要进行修改！")

if __name__ == "__main__":
    main()

import numpy as np

def min2max(X):
    # 极小型转极大型
    return np.max(X) - X

def mid2max(X, best):
    # 中间型转极大型，增加M==0保护
    M = np.max(np.abs(X - best))
    if M == 0:
        # 所有值都等于best，均视为最优
        return np.ones_like(X, dtype=float)
    return 1 - np.abs(X - best) / M

def int2max(X, a, b):
    # 区间型转极大型，增加M==0保护
    M = max(a - np.min(X), np.max(X) - b)
    if M == 0:
        # 全部落在[a,b]或边界，视为最优
        return np.ones_like(X, dtype=float)
    X_new = X.copy().astype(float)
    for i in range(len(X)):
        if X[i] < a:
            X_new[i] = 1 - (a - X[i]) / M
        elif a <= X[i] <= b:
            X_new[i] = 1
        elif X[i] > b:
            X_new[i] = 1 - (X[i] - b) / M
    return X_new

def entropy_weight(X, vec=None, types=None, bests=None, intervals=None):
    '''
    X: np.ndarray or pd.DataFrame, shape (n_samples, n_features)
    vec: list, 需要正向化的列索引（从0开始），如[0,2]
    types: list, 每个正向化列对应的数据类型（1:极小型, 2:中间型, 3:区间型）
    bests: list, 中间型的最好值
    intervals: list, 区间型的[a,b]区间
    '''
    X = X.copy()
    # 支持 DataFrame 输入
    try:
        import pandas as _pd
        if isinstance(X, _pd.DataFrame):
            X = X.values
    except Exception:
        pass

    # 新增：对 vec/types/bests/intervals 长度进行校验与保护
    if vec is not None and len(vec) > 0:
        if types is None or len(types) != len(vec):
            raise ValueError("当指定要正向化的列(vec)时，types 长度必须与 vec 一致。")
        # 若未提供 bests/intervals，则用 None 占位以保证索引对齐
        if bests is None:
            bests = [None] * len(types)
        if intervals is None:
            intervals = [None] * len(types)
        if len(bests) != len(types) or len(intervals) != len(types):
            raise ValueError("bests 和 intervals 的长度应与 types 匹配（按索引对应），缺失请使用 None 占位。")

        for idx, col in enumerate(vec):
            flag = types[idx]
            if flag == 1:
                X[:, col] = min2max(X[:, col])
            elif flag == 2:
                # bests[idx] 已经保证存在（可能为 None 会导致后续函数报错）
                if bests[idx] is None:
                    raise ValueError(f"types 第 {idx} 项为中间型，但 bests[{idx}] 为 None 或缺失。")
                X[:, col] = mid2max(X[:, col], bests[idx])
            elif flag == 3:
                if intervals[idx] is None:
                    raise ValueError(f"types 第 {idx} 项为区间型，但 intervals[{idx}] 为 None 或缺失。")
                a, b = intervals[idx]
                X[:, col] = int2max(X[:, col], a, b)

    # 标准化（增加数值保护）
    X = X.astype(float)
    if np.all(X >= 0):
        # L2 归一化按列
        squere_X = X ** 2
        sum_X = np.sqrt(np.sum(squere_X, axis=0))
        # 防止除以0
        sum_X[sum_X == 0] = 1.0
        stand_X = X / sum_X
    else:
        max_X = np.max(X, axis=0)
        min_X = np.min(X, axis=0)
        range_X = max_X - min_X
        # 防止零区间导致除以0
        range_X[range_X == 0] = 1.0
        stand_X = (X - min_X) / range_X

    # 熵权法（保护列和为0及n<=1情况）
    n, m = stand_X.shape
    col_sums = np.sum(stand_X, axis=0)
    # 防止列和为0
    col_sums[col_sums == 0] = 1e-12
    P = stand_X / col_sums
    # 将概率为0的值替换为小值，避免 log(0)
    eps = 1e-12
    P[P <= 0] = eps
    # 计算信息熵
    H_x = -np.sum(P * np.log(P), axis=0)
    # 当 n=1 时 log(n)=0，需要保护
    den = np.log(n) if n > 1 else 1.0
    e_j = H_x / den
    d_j = 1 - e_j
    # 防止全为0 导致除以0
    if np.sum(d_j) == 0:
        w = np.ones_like(d_j) / len(d_j)
    else:
        w = d_j / np.sum(d_j)
    return w

if __name__ == "__main__":
    # 增加一个交互式入口，行为类似 MATLAB 脚本
    import pandas as pd
    import os

    def _parse_indices(s):
        # 支持格式："-1" 或 "1 2 3" 或 "1,2,3"
        s = s.strip()
        if s == "-1":
            return None
        s = s.replace(',', ' ')
        parts = [p for p in s.split() if p != '']
        # MATLAB 为 1-based 索引，转换为 0-based
        return [int(p) - 1 for p in parts]

    def _parse_interval(s):
        # 支持 "[a,b]" 或 "a b" 或 "a,b"
        s = s.strip().replace('[','').replace(']','').replace(',', ' ')
        parts = [p for p in s.split() if p != '']
        return float(parts[0]), float(parts[1])

    # 读取文件
    excel_path = input("请输入要读取的Excel文件（回车使用 blind date.xlsx）：").strip() or "blind date.xlsx"
    if not os.path.exists(excel_path):
        print(f"未找到文件 {excel_path}，将尝试使用示例数组")
        # 示例数据
        X = np.array([
            [89, 1],
            [60, 3],
            [74, 2],
            [99, 0]
        ], dtype=float)
    else:
        X = pd.read_excel(excel_path).values[:, 1:] #默认跳过第一列

    # 询问是否需要正向化
    vec_in = input('请输入要正向化的向量组，请以空格或逗号分隔，如 "1 2 3"，不需要正向化请输入 -1\n').strip()
    vec = _parse_indices(vec_in)
    types = []
    bests = []
    intervals = []
    if vec is not None:
        for idx in vec:
            flag = int(input(f'第 {idx+1} 列是哪类数据(1:极小型 2:中间型 3:区间型)：\n').strip())
            types.append(flag)
            # 保证 bests 和 intervals 与 types 长度一致：使用 None 占位
            if flag == 2:
                b = float(input('请输入中间型的最好值：\n').strip())
                bests.append(b)
                intervals.append(None)
            elif flag == 3:
                arr = input('请输入最佳区间，按照 "a b" 或 "[a,b]" 的形式输入：\n').strip()
                a, b = _parse_interval(arr)
                intervals.append((a,b))
                bests.append(None)
            else:
                # 极小型：在对应位置放置占位符
                bests.append(None)
                intervals.append(None)

    # 调用计算并输出
    w = entropy_weight(X, vec=vec or [], types=types or [], bests=bests or [], intervals=intervals or [])
    print("熵权为：", w)

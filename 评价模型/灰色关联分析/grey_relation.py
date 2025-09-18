import numpy as np
import pandas as pd
import os

def min2max(X):
    # 极小型转极大型
    return np.max(X) - X

def mid2max(X, best):
    # 中间型转极大型
    M = np.max(np.abs(X - best))
    if M == 0:
        return np.ones_like(X, dtype=float)
    return 1 - np.abs(X - best) / M

def int2max(X, a, b):
    # 区间型转极大型
    M = max(a - np.min(X), np.max(X) - b)
    if M == 0:
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

def grey_relation(X, vec=None, types=None, bests=None, intervals=None, roh=0.5):
    '''
    X: np.ndarray, shape (n_samples, n_features)
    vec: list, 需要正向化的列索引（从0开始）
    types: list, 每个正向化列的数据类型（1:极小型, 2:中间型, 3:区间型）
    bests: list, 中间型的最好值
    intervals: list, 区间型的[a,b]区间
    roh: 分辨系数，默认0.5
    '''
    X = X.copy()
    # 正向化
    if vec is not None and len(vec) > 0:
        if types is None or len(types) != len(vec):
            raise ValueError("types 长度必须与 vec 一致")
        if bests is None:
            bests = [None] * len(types)
        if intervals is None:
            intervals = [None] * len(types)
        for idx, col in enumerate(vec):
            flag = types[idx]
            if flag == 1:
                X[:, col] = min2max(X[:, col])
            elif flag == 2:
                if bests[idx] is None:
                    raise ValueError(f"types 第 {idx} 项为中间型，但 bests[{idx}] 为 None。")
                X[:, col] = mid2max(X[:, col], bests[idx])
            elif flag == 3:
                if intervals[idx] is None:
                    raise ValueError(f"types 第 {idx} 项为区间型，但 intervals[{idx}] 为 None。")
                a, b = intervals[idx]
                X[:, col] = int2max(X[:, col], a, b)

    # 标准化
    n, m = X.shape
    if np.all(X >= 0):
        squere_X = X ** 2
        sum_X = np.sqrt(np.sum(squere_X, axis=0))
        sum_X[sum_X == 0] = 1.0
        stand_X = X / sum_X
    else:
        max_X = np.max(X, axis=0)
        min_X = np.min(X, axis=0)
        range_X = max_X - min_X
        range_X[range_X == 0] = 1.0
        stand_X = (X - min_X) / range_X

    # 灰色关联分析
    res = stand_X
    x0 = np.max(res, axis=1, keepdims=True)  # 参考序列，按行最大
    gre_X = np.hstack([res, x0])
    m, n = gre_X.shape
    gamma_X = np.abs(gre_X[:, :-1] - gre_X[:, -1][:, None])
    a = np.min(gamma_X)
    b = np.max(gamma_X)
    gamma = (a + roh * b) / (gamma_X + roh * b)
    gre_res = np.sum(gamma, axis=0) / m
    return gre_res

if __name__ == "__main__":
    # 交互式入口
    def _parse_indices(s):
        s = s.strip()
        if s == "-1":
            return None
        s = s.replace(',', ' ')
        parts = [p for p in s.split() if p != '']
        return [int(p) - 1 for p in parts]

    def _parse_interval(s):
        s = s.strip().replace('[','').replace(']','').replace(',', ' ')
        parts = [p for p in s.split() if p != '']
        return float(parts[0]), float(parts[1])

    # 读取文件
    excel_path = input("请输入要读取的Excel文件（回车使用 data.xlsx）：").strip() or "data.xlsx"
    if not os.path.exists(excel_path):
        print(f"未找到文件 {excel_path}，将尝试使用示例数组")
        X = np.array([
            [89, 1],
            [60, 3],
            [74, 2],
            [99, 0]
        ], dtype=float)
    else:
        X = pd.read_excel(excel_path).values

    # 正向化
    vec_in = input('请输入要正向化的向量组，请以空格或逗号分隔，如 "1 2 3"，不需要正向化请输入 -1\n').strip()
    vec = _parse_indices(vec_in)
    types = []
    bests = []
    intervals = []
    if vec is not None:
        for idx in vec:
            flag = int(input(f'第 {idx+1} 列是哪类数据(1:极小型 2:中间型 3:区间型)：\n').strip())
            types.append(flag)
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
                bests.append(None)
                intervals.append(None)
    res = grey_relation(X, vec=vec, types=types, bests=bests, intervals=intervals, roh=0.5)
    print("灰色关联度结果：", res)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class KnapsackResult:
    max_value: int
    selected_indices: List[int]  # 选中的物品索引
    selected_weights: List[int]
    selected_values: List[int]

def knapsack_2d(weights: List[int], values: List[int], capacity: int) -> KnapsackResult:
    """
    二维 DP 求解 0-1 背包，并恢复物品选择方案。
    时间: O(n*C) 空间: O(n*C)
    """
    n = len(weights)
    if n != len(values):
        raise ValueError("weights 与 values 长度不一致")
    # dp[i][c] 前 i 个物品在容量 c 下最大价值
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        w, val = weights[i-1], values[i-1]
        for c in range(capacity+1):
            dp[i][c] = dp[i-1][c]
            if c >= w:
                cand = dp[i-1][c-w] + val
                if cand > dp[i][c]:
                    dp[i][c] = cand
    # 方案恢复
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i-1][c]:
            selected.append(i-1)
            c -= weights[i-1]
    selected.reverse()
    return KnapsackResult(
        max_value=dp[n][capacity],
        selected_indices=selected,
        selected_weights=[weights[i] for i in selected],
        selected_values=[values[i] for i in selected],
    )

def knapsack_1d(weights: List[int], values: List[int], capacity: int) -> KnapsackResult:
    """
    一维滚动数组版本，时间 O(n*C)，空间 O(C)。
    通过额外记录选择路径来恢复方案。
    """
    n = len(weights)
    if n != len(values):
        raise ValueError("weights 与 values 长度不一致")

    dp = [0]*(capacity+1)
    # choice[i][c] = True 表示物品 i 在最优解中使用（针对该层）
    choice = [[False]*(capacity+1) for _ in range(n)]
    for i, (w, val) in enumerate(zip(weights, values)):
        for c in range(capacity, w-1, -1):
            cand = dp[c-w] + val
            if cand > dp[c]:
                dp[c] = cand
                choice[i][c] = True
    # 方案恢复
    selected = []
    c = capacity
    for i in range(n-1, -1, -1):
        if c >= 0 and choice[i][c]:
            selected.append(i)
            c -= weights[i]
    selected.reverse()
    return KnapsackResult(
        max_value=dp[capacity],
        selected_indices=selected,
        selected_weights=[weights[i] for i in selected],
        selected_values=[values[i] for i in selected],
    )

def knapsack(weights: List[int], values: List[int], capacity: int, method: str = "1d") -> KnapsackResult:
    """
    method: "1d" 或 "2d"
    """
    if method == "1d":
        return knapsack_1d(weights, values, capacity)
    elif method == "2d":
        return knapsack_2d(weights, values, capacity)
    else:
        raise ValueError("method 只能为 '1d' 或 '2d'")

def _demo():
    weights = [2, 2, 6, 5, 4]
    values  = [6, 3, 5, 4, 6]
    capacity = 10
    for m in ("2d", "1d"):
        res = knapsack(weights, values, capacity, method=m)
        print(f"方法: {m}")
        print("最大价值:", res.max_value)
        print("选中索引:", res.selected_indices)
        print("选中重量:", res.selected_weights)
        print("选中价值:", res.selected_values)
        print("-"*40)

if __name__ == "__main__":
    _demo()

import random
import math
import networkx as nx  # 新增: 引入 networkx

# 成本矩阵，999 代表不可达（对应 Matlab 版本）
C = [
    [0,   50, 999, 40, 25, 10],
    [50,   0,  15, 20, 999, 25],
    [999, 15,  0,  10, 20, 999],
    [40,  20, 10,  0,  10, 25],
    [25, 999, 20, 10,  0,  55],
    [10,  25, 999,25, 55,  0 ]
]

def random_simple_path(C, src, dst, max_len):
    """随机生成一条简单路径（不保证最优），返回距离；不可达返回 inf。"""
    if src == dst:
        return 0
    n = len(C)
    visited = {src}
    current = src
    dist = 0
    for _ in range(max_len):
        if current == dst:
            return dist
        neighbors = [k for k in range(n) if C[current][k] < 999 and k not in visited]
        if not neighbors:
            break
        nxt = random.choice(neighbors)
        dist += C[current][nxt]
        visited.add(nxt)
        current = nxt
    if current == dst:
        return dist
    return math.inf

def monte_carlo_shortest(C, src=0, iters=10000):
    """Monte Carlo 近似最短路：多次随机采样取最小值。"""
    n = len(C)
    best = [math.inf] * n
    best[src] = 0
    for _ in range(iters):
        for target in range(n):
            if target == src:
                continue
            d = random_simple_path(C, src, target, max_len=n-1)
            if d < best[target]:
                best[target] = d
    return best

def dijkstra(C, src=0):
    """标准 Dijkstra（适用于非负权）。"""
    n = len(C)
    dist = [math.inf] * n
    prev = [None] * n
    visited = [False] * n
    dist[src] = 0
    for _ in range(n):
        # 选取未访问中距离最小的点
        u = None
        best = math.inf
        for i in range(n):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u is None:
            break
        visited[u] = True
        for v in range(n):
            if C[u][v] < 999 and not visited[v]:
                alt = dist[u] + C[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    return dist, prev

def networkx_dijkstra(C, src=0):
    """使用 networkx 复现 Matlab shortestpath 行为（无向加权图）。"""
    G = nx.Graph()
    n = len(C)
    for i in range(n):
        for j in range(i + 1, n):
            w = C[i][j]
            if w < 999:  # 999 视为不可达
                G.add_edge(i, j, weight=w)
    dist, paths = nx.single_source_dijkstra(G, src)  # 所有点最短路
    return dist, paths

def reconstruct_path(prev, target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

def main():
    random.seed(0)  # 可复现
    src = 0  # Matlab 中节点 1，这里用 0 表示
    # Monte Carlo 近似
    mc_result = monte_carlo_shortest(C, src=src, iters=10000)
    # Dijkstra 精确
    dj_dist, prev = dijkstra(C, src=src)
    target = 3  # Matlab shortestpath(G,1,4)
    path = reconstruct_path(prev, target)
    # 输出
    print("Monte Carlo 近似距离(源点=1):")
    for i, d in enumerate(mc_result):
        print(f"  1 -> {i+1}: {d if d < math.inf else '不可达'}")
    print("\nDijkstra 精确距离(源点=1):")
    for i, d in enumerate(dj_dist):
        print(f"  1 -> {i+1}: {d if d < math.inf else '不可达'}")
    # 还原为 1-based 路径
    path_1_based = [p + 1 for p in path]
    print(f"\nDijkstra 1->4 最短路径: {path_1_based}, 距离: {dj_dist[target]}")
    print("\n说明: Monte Carlo 结果为采样近似，仅作演示；Dijkstra 为精确最短路.")
    # --- 新增 NetworkX 示例 ---
    nx_dist, nx_paths = networkx_dijkstra(C, src=src)
    nx_path_1_4 = [p + 1 for p in nx_paths[target]]
    print("\nNetworkX Dijkstra 1->4 最短路径:", nx_path_1_4, "距离:", nx_dist[target])
    print("NetworkX 结果应与自实现 Dijkstra 一致。")

if __name__ == "__main__":
    main()

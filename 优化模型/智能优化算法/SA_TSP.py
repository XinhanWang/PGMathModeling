import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_chinese_font():
    """
    返回指定路径的微软雅黑字体 FontProperties 实例。
    """
    return FontProperties(fname='/home/xhwang/msyh.ttc')

plt.rcParams['axes.unicode_minus'] = False
chinese_font = get_chinese_font()

def caculate_tsp(path, dist):
    return sum(dist[path[i], path[i+1]] for i in range(len(path)-1))

def gen_new_path(path0, dist):
    # 改良圈算法（2-opt）
    L = len(path0)
    path1 = path0.copy()
    for i in range(L-2):
        for j in range(i+2, L-1):
            a, b = path1[i], path1[i+1]
            c, d = path1[j], path1[j+1]
            if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d]:
                path1[i+1:j+1] = path1[i+1:j+1][::-1]
    return path1

def gen_new_path1(path0):
    # 交换法
    n = len(path0)
    c1 = np.random.randint(1, n-1)
    c2 = np.random.randint(1, n-1)
    while c1 == c2:
        c2 = np.random.randint(1, n-1)
    path1 = path0.copy()
    path1[c1], path1[c2] = path1[c2], path1[c1]
    return path1

def main():
    np.random.seed(0)
    # 加载城市坐标
    # coord = np.load('coord_mat.npy')  # 假设已保存为numpy格式
    coord = np.array([
    [6734, 1453],
    [2233, 10],
    [5530, 1424],
    [401, 841],
    [3082, 1644],
    [7608, 4458],
    [7573, 3716],
    [7265, 1268],
    [6898, 1885],
    [1112, 2049],
    [5468, 2606],
    [5989, 2873],
    [4706, 2674],
    [4612, 2035],
    [6347, 2683],
    [6107, 669],
    [7611, 5184],
    [7462, 3590],
    [7732, 4723],
    [5900, 3561],
    [4483, 3369],
    [6101, 1110],
    [5199, 2182],
    [1633, 2809],
    [4307, 2322],
    [675, 1006],
    [7555, 4819],
    [7541, 3981],
    [3177, 756],
    [7352, 4506],
    [7545, 2801],
    [3245, 3305],
    [6426, 3173],
    [4608, 1198],
    [23, 2216],
    [7248, 3779],
    [7762, 4595],
    [7392, 2244],
    [3484, 2829],
    [6271, 2135],
    [4985, 140],
    [1916, 1569],
    [7280, 4899],
    [7509, 3239],
    [10, 2676],
    [6807, 2993],
    [5185, 3258],
    [3023, 1942],
], dtype=int)
    
    n = coord.shape[0]
    begin_city = 0
    end_city = 1
    tmp = coord.copy()
    coord = np.delete(coord, [begin_city, end_city], axis=0)
    coord = np.vstack([tmp[begin_city], coord, tmp[end_city]])

    # 绘制城市分布
    plt.figure()
    plt.plot(coord[:,0], coord[:,1], 'o')
    plt.title('City Distribution', fontproperties=chinese_font)
    plt.show(block=False)

    # 距离矩阵
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coord[i, :2] - coord[j, :2])

    # 模拟退火参数
    T0 = 1000
    T = T0
    maxgen = 200
    Lk = 20
    alfa = 0.95

    # 初始解
    path0 = np.random.permutation(np.arange(1, n-1)) 
    path0 = np.concatenate(([0], path0, [n-1]))
    path0 = gen_new_path(path0, dist)  # 使用改良圈算法优化初始解
    result0 = caculate_tsp(path0, dist)
    min_result = result0
    best_path = path0.copy()
    RESULT = []

    for iter in range(maxgen):
        for _ in range(Lk):
            # path1 = gen_new_path1(path0)
            path1 = gen_new_path(path0, dist)
            result1 = caculate_tsp(path1, dist)
            if result1 < result0:
                path0 = path1
                result0 = result1
            else:
                p = np.exp(-(result1 - result0)/T)
                if np.random.rand() < p:
                    path0 = path1
                    result0 = result1
            if result0 < min_result:
                min_result = result0
                best_path = path0.copy()
        RESULT.append(min_result)
        T *= alfa

    # 打印路径
    path_str = '-->'.join(str(i) for i in best_path)
    print('最佳的方案是：', path_str)
    print('对应最优值是：', min_result)

    # 绘制最优路径
    plt.figure()
    plt.plot(coord[:,0], coord[:,1], 'o')
    for i in range(n-1):
        x = [coord[best_path[i],0], coord[best_path[i+1],0]]
        y = [coord[best_path[i],1], coord[best_path[i+1],1]]
        plt.plot(x, y, '-b')
    plt.title('Best Path', fontproperties=chinese_font)
    plt.show()

    # 迭代曲线
    plt.figure()
    plt.plot(range(1, maxgen+1), RESULT, 'b-')
    plt.xlabel('迭代次数', fontproperties=chinese_font)
    plt.ylabel('路径长度', fontproperties=chinese_font)
    plt.title('SA Progress', fontproperties=chinese_font)
    plt.show()

if __name__ == '__main__':
    main()

import numpy as np

# 距离矩阵
A = np.array([
    [0, 56, 21, 35],
    [56, 0, 49, 39],
    [21, 49, 0, 77],
    [35, 39, 77, 0]
])

L = A.shape[0]
c = [0, 1, 2, 3, 0]  # Python索引从0开始

for k in range(L):
    flag = 0
    for i in range(L - 2):
        for j in range(i + 2, L):
            # 判断是否可以通过交换获得更短路径
            if (A[c[i], c[j]] + A[c[i+1], c[(j+1)%L]] < 
                A[c[i], c[i+1]] + A[c[j], c[(j+1)%L]]):
                # 翻转路径
                c[i+1:j+1] = c[j:i:-1]
                flag += 1
    if flag == 0:
        long = 0
        for i in range(L):
            long += A[c[i], c[i+1]]
        print("最短圈长:", long)
        print("最优路径:", [x+1 for x in c])  # 输出为1-based索引
        break

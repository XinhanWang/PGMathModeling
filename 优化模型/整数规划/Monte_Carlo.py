import numpy as np
import matplotlib.pyplot as plt

a = 10
L = 5
n = 100000  # 抛掷次数
np.random.seed(0)
# 随机抛针，得到的角度
ph = np.random.rand(n) * np.pi
x = np.random.rand(n) * a / 2
y = (L / 2) * np.sin(ph)  # 恰好相交的边界情况

m = 0  # 相交次数

plt.axis([0, np.pi, 0, a / 2])
plt.box(True)

for i in range(n):
    if x[i] <= y[i]:
        m += 1
        plt.plot(ph[i], x[i], 'b.')

P = m / n
mypi = 2 * L / (P * a)
print("估算的π值为:", mypi)

plt.xlabel('ph')
plt.ylabel('x')
plt.title('Monte Carlo Simulation (Buffon Needle)')
plt.show()

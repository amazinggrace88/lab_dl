import numpy as np

x0 = np.arange(-2, 3)
x1 = np.arange(-2, 3)
print(x0)
print(x1)
XY = np.array([x0, x1])
print(XY)

for i, x_i in enumerate(XY):
    print(i)  # index 출력
    print(x_i)  # 데이터 출력 - 2차원이면 [] 리스트 출력


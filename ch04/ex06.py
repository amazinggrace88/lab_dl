"""
?? idk
"""
import numpy as np
from lab_dl.ch04.ex05 import numerical_gradient
import matplotlib.pyplot as plt


def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)  # 1행 sum(x**2)으로 (행, 열)


x0 = np.arange(-1, 2)  # -2~(3-1) 범위로 np.array
print('x0 = ', x0)
x1 = np.arange(-1, 2)
print('x1 = ', x1)

# meshgrid 의 개념
X, Y = np.meshgrid(x0, x1)  # Return coordinate matrices from coordinate vectors. x0행, x1열인 2차원 행렬
print('X = ', X)
print('Y = ', Y)


# meshgrid 의 사용
X = X.flatten()  # why? - numerical_gradient 에 넣을 parameter x 를 만들기 위해서
Y = Y.flatten()
print(X)
print(Y)
XY = np.array([X, Y])  # (x, y) 좌표를 만들었다. - np.array 모양 (조합)
print('XY = ', XY)

gradients = numerical_gradient(fn, XY)
print('gradient = ', gradients)


# 수정된 x0, x1 로 gradient 구하기
x0 = np.arange(-2, 2.5, 0.25)
print('수정된 x0 = ', x0)
x1 = np.arange(-2, 2.5, 0.25)
print('수정된 x1 = ', x1)
X, Y = np.meshgrid(x0, x1)  # meshgrid 는 X, Y 로 두어야 해~ 조심~
X = X.flatten()
Y = Y.flatten()
XY = np.array([X, Y])
gradients = numerical_gradient(fn, XY)
print('gradient = ', gradients)
plt.quiver(X, Y, -gradients[0], -gradients[1], angles='xy')
# X - 1차원 배열
# Y - 1차원 배열
# -gradients[0] - x 축 좌표
# -gradients[1] - y 축 좌표
plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Gradient \'s direction')
plt.show()
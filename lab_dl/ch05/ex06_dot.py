"""
dot
"""
import numpy as np
np.random.seed(103)
X = np.random.randint(10, size=(2, 3))
print('X = \n', X)  # (2, 3) array
W = np.random.randint(10, size=(3, 5))
print('W = \n', W)  # (3, 5) array
# forward propagation
Z = X.dot(W)  # Z : (2, 5) array
print('Z = \n', Z)
# back propagation
delta = np.random.randn(2, 5)  # 파라미터 순서 잘 보기 = tuple 안된다.
print('delta = ', delta)  # random 숫자를 만들었다.

# X 방향으로의 오차 역전파 만들기
dX = delta.dot(W.T)  # W.T 는 transpose - X의 모양과 같아지기 위해서, (2, 5) @ (3, 5)T = (2, 5) @ (5, 3) = (2, 3)
print('dx = ', dX)
# W 방향으로의 오차 역전파
dW = (X.T).dot(delta)
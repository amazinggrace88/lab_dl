"""
선형대수 복습
- 다차원 배열의 계산
밑바닥부터 시작하는 딥러닝 p.77
"""

"""
행렬의 내적(dot product)
A, B, C, D.. : 대문자로 선언하는 변수들 - 2차원 이상의 ndarray
x, y ...     : 1차원 이상의 ndarray
"""
import numpy as np


# 행렬 만드는 방법 1
x = np.array([1, 2])  # 1차원 (1x2)
W = np.array([[3, 4],
              [5, 6]])  # 2차원 (2x2)
print('x dot W : \n', x.dot(W))
print('W dot x : \n', W.dot(x))  # numpy 가 알아서 해주는 거~ 원래 안된다잉 (한쪽이 1차원이면 알아서 모양을 바꿔서 계산해줌) 주의하기

# * (2x2)는 모양을 바꾸지 않지만,
# 1차원 array 는 (1x2)을 (2x1)로 서로서로 바꾸기도 한다.


# 행렬 만드는 방법 2
A = np.arange(1, 7).reshape((2, 3))
print('A :\n', A)
B = np.arange(1, 7).reshape((3, 2))
print('B :\n', B)
print('A dot B : \n', A.dot(B))
print('B dot A : \n', B.dot(A))  # 교환법칙이 성립하지 않음


# 차원에 따른 ndarray.shape
# (x, ), (x, y), (x, y, z)...
x = np.array([1, 2, 3])
print(x)
print(x.shape)
# 원소 3개 : (x, ) 행벡터로 변환 [1, 2, 3], 열벡터로 변환 가능! (shape 지정하지 않아서)

x = x.reshape((3, 1))  # shape 지정, 2차원 배열이 되면서 열벡터가 되었다.
# x = x.reshape(3, 1) 괄호 1개만 써도 똑같이 나옴 (그래도 괄호 두개 쓰장)
print(x)
# [[1]
#  [2]
#  [3]]
print(x.shape)  # (3, 1)

x = x.reshape((1, 3))
print(x)
# [[1 2 3]]
print(x.shape)  # (1, 3)
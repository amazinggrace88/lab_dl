"""
신경망을 행렬로 나타내기
"""
import numpy as np

x = np.array([1, 2])
W = np.array([[1, 2],
              [3, 4]])
print('x @ W', x.dot(W))
print('W @ x', W.dot(x))  # numpy 의 기능

C = np.arange(1, 7).reshape((2, 3))
print('sample arange : \n', C)
D = np.arange(1, 10).reshape((3, 3))
print('sample arange2 : \n', D)
E = np.array([1, 2, 3])
print('sample array1 : \n', E.reshape((3, 1)))
print(E.shape)


def step_function(x):
    result = [1 if x_i > 0 else 0
              for x_i in x]
    return result



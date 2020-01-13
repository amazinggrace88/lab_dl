"""
Deep learning from scratch p.82 그림 3-14 행렬로 구현
"""
import numpy as np
x = np.array([1, 2])
print(x.shape)  # 원소의 갯수 2개인 행벡터
W = np.array([[1, 3, 5],
              [2, 4, 6]])
print(W.shape)
y = x.dot(W) + 1  # bias 는 각 배열 + 1 -> numpy broadcasting
print(y, y.shape)  # 원소의 갯수 3개인 행벡터

# 방법 2
W2 = np.array([[1, 2],
               [3, 4],
               [5, 6]])
print(W2.shape)
y2 = W2.dot(x) + 1
print(y2, y2.shape)  # 원소의 갯수 3개인 열벡터 -> numpy broadcasting


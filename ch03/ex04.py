"""
Deep learning from scratch p.85 그림 3-17 행렬로 구현
- 신경망은 행렬로 가장 잘 나타낼 수 있어~!
목적 : 행렬식 중 가장 오차가 적은 w1, w2 .. 를 찾는 것이다.

"""
import numpy as np
from ch03.ex01 import sigmoid

# a(1) : 첫번째 은닉층 만들기 그림 3-17 (식 3.9)
# a1 = x @ W1 + b1
x = np.array([1, 2])
# x1, x2 = 1, 2 라고 숫자를 주자
W1 = np.array([[1, 2, 3],
              [4, 5, 6]])
b1 = np.array([1, 2, 3])
a1 = x.dot(W1) + b1
print('a(1) layer : ', a1)

# 출력 a(1) 에 활성화 함수를 sigmoid 함수로 적용 (그림 3-18)
# z1 = sigmoid(a1) - 숫자를 0~1사이의 숫자로 바꿔준다.
z1 = sigmoid(a1)
print('sigmoid function list of a(1) :', z1)

# a(2) : 두번째 은닉층 만들기 그림 3-19
# a2 = z1 @ W2 + b2
W2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
a2 = z1.dot(W2) + b2
print('a(2) layer : ', a2)

# a(2) 에 활성화 함수(sigmoid) 적용
z2 = sigmoid(a2)
print('sigmoid function list of a(2) :', z2)


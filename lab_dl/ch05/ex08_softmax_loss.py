"""
softmax_loss 계층 함수 만들기 p.176
"""
from lab_dl.ch03.ex11 import softmax
from lab_dl.ch04.ex03 import cross_entropy
import numpy as np


class SoftmaxWithLoss:
    def __init__(self):
        self.y_true = None  # 정답 레이블을 저장하기 위한 field(변수), one-hot-encoding 가정
        self.y_pred = None  # softmax 함수 출력 (예측 레이블) 을 저장하기 위한 field
        self.loss = None  # cross_entropy 함수의 출력 (손실, 오차) 를 저장하기 위한 field

    def forward(self, X, Y_true):  # 행렬 X
        self.y_true = Y_true
        self.y_pred = softmax(X)  # softmax 에서 나오는 출력들
        loss = cross_entropy(self.y_pred, self.y_true)
        return loss

    def backward(self, dout=1):
        if self.y_true.ndim == 1:
            n = 1
        else:
            n = self.y_true.shape[0]  # y_true 의 row 의 갯수 - 1 차원일 때
        dx = (self.y_pred - self.y_true) / n  # error 들의 평균 (수학적 증명 이해 필요)
        return dx


if __name__ == '__main__':
    np.random.seed(103)
    x = np.random.randint(10, size=3)
    print('x = ', x)  # a1, a2, a3
    # case 1
    y_true = np.array([1., 0., 0.])  # one_hot_encoding
    print('y true = ', y_true)

    swl = SoftmaxWithLoss()  # 생성자 생성
    loss = swl.forward(x, y_true)  # forward propagation
    print('loss = ', loss)  # 손실 = 오차들의 평균
    print('y_pred = ', swl.y_pred)
    dx = swl.backward()  # back propagation - 변화율
    print('dx = ', dx)
    print()

    # case 2 손실이 가장 큰 경우
    y_true = np.array([0., 0., 1.])
    print('y true = ', y_true)
    loss = swl.forward(x, y_true)
    print('loss = ', loss)  # 손실 = 오차들의 평균
    print('y_pred = ', swl.y_pred)
    dx = swl.backward()  # back propagation
    print('dx = ', dx)
    print()

    # case 3 손실이 가장 작은 경우
    y_true = np.array([0., 1., 0.])
    print('y true = ', y_true)
    loss = swl.forward(x, y_true)
    print('loss = ', loss)  # 손실 = 오차들의 평균
    print('y_pred = ', swl.y_pred)
    dx = swl.backward()  # back propagation
    print('dx = ', dx)

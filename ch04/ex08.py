"""
신경망에서 gradient 기울기 descent 하강 방법

(구성)
신경망 하나 (뉴런 3개)
입력값 2개 x1, x2(w1, w1)

- y_true, y_pred 를 계산하고 softmax의 함수를 계산하여 cross entropy 의 값을 줄여준다.
"""
import numpy as np

from lab_dl.ch03.ex11 import softmax
from lab_dl.ch04.ex03 import cross_entropy
from lab_dl.ch04.ex05 import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        # 가중치 행렬의 초깃값들을 임의로 설정
        self.W = np.random.randn(2, 3)  # 2행 3열 - 숫자 6개의 평균 0, 표준편차 1인 표준정규분포

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, y_true):
        """
        손실함수 (loss function) - cross entropy
        x - train data
        y_true - label
        """
        y_pred = self.predict(x)
        ce = cross_entropy(y_pred, y_true)  # train data 에서 y_pred 구해서 ce 구했음
        return ce

    def gradient(self, x, t):  # x - train data , t - y_true 의미
        fn = lambda W: self.loss(x, t)  # fn 의 손실함수를 구했음 - ce 나옴. W 이용 안되도 사용 가능
        return numerical_gradient(fn, self.W)  # p1 에 대한 편미분을 계산함


if __name__ == '__main__':
    network = SimpleNetwork()  # 클래스 생성자 호출 -> __init__() 메소드 호출
    print('W = ', network.W)  # 임의로 생성된 W

    # 데이터 설정 : x = [0.6, 0.9] 일 때 y_true = [0, 0, 1] 라고 가정
    x = np.array([0.6, 0.9])
    y_true = np.array([0.0, 0.0, 1.0])
    print('x = ', x)
    print('y_true = ', y_true)

    # 예측 : gradient descent 과정 전 (임의의 W 적용)
    y_pred = network.predict(x)
    print('y pred = ', y_pred)

    # 경사하강법의 기준 : y_pred - y_true 의 오차인 cross entropy
    ce = network.loss(x, y_true)
    print('cross entropy = ', ce)  # ce 를 감소시키는 것 목적!

    # 경사하강법을 이용하여 기울기 구하기 - 문제가 생김
    g1 = network.gradient(x, y_true)  # 기울기
    print('gradient decent 1 = \n', g1)

    lr = 0.1
    network.W -= lr * g1
    print('W = ', network.W)  # 바뀐 것을 볼 수 있음
    print('y_pred = ', network.predict(x))
    print('cross entropy after gd = ', network.loss(x, y_true))
    print()

    # for 문 안에서 100번 반복
    lr = 0.1
    for i in range(100):
        g1 = network.gradient(x, y_true)  # 기울기
        # print('for : gradient decent 1 = \n', g1)
        network.W -= lr * g1
        # print('for : W = ', network.W)  # 바뀐 것을 볼 수 있음
        print(f'for : y_pred = {network.predict(x)}', f'for t: cross entropy after gd = {network.loss(x, y_true)}')


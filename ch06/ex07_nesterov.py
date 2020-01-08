"""
파라미터 최적화 알고리즘
5) nesterov
1. nesterov 개념 설명
Momentum : v 라는 상수가 있어
           v_1 = m * v_0 - lr * dL/dW
           W_1 = W_0 + v_1
               = W_0 + m * v_0 - lr * dL/dW  (m * v 을 더해준 것뿐)
           W_1 식에 들어가는 v 는 변경 전에 초기값 v_0이다.

Nesterov : v_1 을 W_1 식에 넣어준다.
           W_1 = W_0 + m * v_1 - lr * dL/dW
               = W_0 + m * (m * v_0 - lr * dL/dW) - lr * dL/dW
               = W_0 + m ** 2 * v_0 - (1 + m) * lr * dL/dW
           W_1 식에 들어가는 v 는 변경 후 v_1이다.
           (1 + m) 으로 가중치 변경이 커지도록 함
           m ** 2 * v_0
           쉽게 생각하면, 경사면을 따라서 떨어질 때 더 빠른 속도로 떨어지게 한다.
"""
import numpy as np
import matplotlib.pyplot as plt
from lab_dl.ch06.ex01_matplot3d import fn_derivative, fn


class Nesterov:
    def __init__(self, lr=0.1, m=0.9):
        self.lr = lr
        self.m = m
        self.v = dict()  # momentum

    def update(self, params, gradients):
        # v = m * v - lr * dL/dW
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])
        for key in params:
            # v = m * v - lr * dL/dW
            self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
            # W_1 = W_0 + m * v_1 - lr * dL/dW
            # W_1 = W_0 + m ** 2 * v_0 - (1 + m) * lr * dL/dW..??
            params[key] += self.m * self.m * self.v[key] - self.lr * gradients[key]


if __name__ == '__main__':
    # momentum 클래스 인스턴스 생성 / params, gradients 초기값 생성
    nestrov = Nesterov()
    params = {'x': -7.0, 'y': 2.0}  # 실수로 바꿔주어야 update 실행 가능 (line 40)
    gradients = {'x': 0, 'y': 0}

    # 파라미터 갱신 값들을 저장할 리스트
    x_history = []
    y_history = []

    # update
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])  # 신경망의 def.gradient 계산 부분
        nestrov.update(params, gradients)

    # 점의 이동 방향 보기
    for x, y in zip(x_history, y_history):
        print(f'{x}, {y}')


    # contour 그래프에 파라미터 갱신 값 그래프 추가
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nesterov')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()
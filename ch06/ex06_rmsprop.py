"""
파라미터 최적화 알고리즘
4) RMSProp
1. RMSProp 개념 설명
SGD : W = W - lr * dL/dW
    단점) lr(학습률) 을 학습하는 동안 변경할 수 없다(epoch 1번 모두 끝날 때까지 변경 불가능)
Adagrad : W = W - (lr/sqrt(h)) * dL/dW
          h = h + (dL/dW) ** 2
    SGD 보완하는 장점) 학습률을 학습하는 동안 변경 가능.
    단점) 학습을 오래 하게 되면 어느 순간 h 가 굉장히 커지는 순간 발생 -> 갱신되는 양이 0 이 되어 학습이 되지 않는 경우 발생 가능

Adagrad 의 단점인 갱신량이 0이 되는 문제를 해결하기 위한 알고리즘 : RMSProp

2. 알고리즘 설명
RMSProp : h 를 rho 를 포함한 식으로 바꾸었다.
rho : (decay-rate 감쇠율) 을 표현하는 하이퍼 파라미터
h = rho * h + (1 - rho) * (dL/dW) ** 2
    rho 가 1에 가까워질수록 (dL/dW) ** 2 이 어려워 h 증가가 많이 커지지 않는다.
    즉, rho 는 gradient(dL/dW) 의 제곱이 무한대로 가는 것을 방지한다.
"""
import numpy as np
from lab_dl.ch06.ex01_matplot3d import fn, fn_derivative

class RMSProp:
    def __init__(self, lr=0.01, rho=0.1):
        self.lr = lr
        self.rho = rho
        self.h = dict()

    def update(self, params, gradients):
        # W = W - (lr/sqrt(h)) * (dL/dW)
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])
        for key in params:
            # h = rho * h + (1 - rho) * (dL/dW) ** 2
            epsilon = 1e-8
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * gradients[key] * gradients[key]
            params[key] -= (self.lr / np.sqrt(self.h[key] + epsilon)) * gradients[key]


if __name__ == '__main__':
    params = {'x': -7.0, 'y': 2.0}
    gradients = {'x': 0, 'y': 0}
    rmsprop = RMSProp()

    x_history = []
    y_history = []

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        rmsprop.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'{x}, {y}')
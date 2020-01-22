"""
**파라미터 최적화 알고리즘**
4) adam (Adaptive (gradient) Moment estimate)
1. adam 개념 설명
Adagrad + Momentum 이 결합
학습률 변화 + 속도(momentum) 개념 도입

2. 알고리즘 설명
W : 파라미터
t (timestamp) : 반복할 때마다 증가하는 숫자.
                시간이 지날 때마다 t 증가
                update 메소드가 호출될 때마다 +1
m (momentum) : 첫번째 모멘텀 first momentum
               ~ gradient(dL/dW) 즉, gradient 에 비례 -> SGD 의 gradient 를 수정함.
               m = beta1_일어날 확률 * m  + (1 - beta1)_일어나지 않을 확률 * grad
v : 두번째 모멘텀 second momentum
    ~ gradient**2(dL/dW) 즉, gradient 제곱에 비례 -> SGD 의 학습률을 수정함.
    v = beta2 * v + (1 - beta2) * grad * grad
beta1, beta2 : m , v 변화에 사용되는 상수 (범위 0~1) -> ?
lr : 학습률(learning rate)

m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
W = W - [lr / sqrt(v_hat)] * m_hat
    (m_hat 이 gradient 역할)
    ([lr / sqrt(v_hat)] 이 lr 역할)
"""
import numpy as np
import matplotlib.pyplot as plt
from lab_dl.lab_dl.ch06.ex01_matplot3d import fn, fn_derivative


class Adam:
    def __init__(self, lr=0.01,  beta1=0.9, beta2=0.99):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0  # 1씩 증가
        self.m = dict()  # 파라미터 종류에 따라서 만들어지는 모양이 달라지기 때문에 dict()로 만든다.
        self.v = dict()

    def update(self, params, gradients):
        self.t += 1  # update 호출될 때마다 timestamp 1씩 증가
        if not self.m:  # m 이 비어있는 딕셔너리 (m이 원소가 없으면 m을 기반으로 하는 v 도 원소가 하나도 없다)
            for key in params:
                # m, v momentum 을 parameter 의 shape 과 동일하게 생성
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        epsilon = 1e-8  # 0 으로 나누는 경우를 방지하기 위해 사용할 상수
        for key in params:
            # m = beta1_일어날 확률 * m  + (1 - beta1)_일어나지 않을 확률 * grad
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            # v =  beta2 * v + (1 - beta2) * grad * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * gradients[key]**2
            # m_hat = m / (1 - beta1 ** t)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # v_hat = v / (1 - beta2 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # W = W - [lr / sqrt(v_hat)] * m_hat
            params[key] -= (self.lr / (np.sqrt(v_hat) + epsilon)) * m_hat


if __name__ == '__main__':
    params = {'x': -7.0, 'y': 2.0}
    gradients = {'x': 0.0, 'y': 0.0}  # 어차피 변경될 것이므로 상관없음!

    # Adam 클래스의 인스턴스 생성
    adam = Adam(lr=0.29)

    # 학습하면서 파라미터(x, y)들이 업데이트 되는 내용을 저장하기 위한 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        adam.update(params, gradients)
        # 변경된 파라미터 값 출력
        print(f"({params['x']}, {params['y']})")

    # 등고선(contour) 그래프
    x = np.linspace(-10, 10, 2000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 8
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam')
    plt.axis('equal')
    # x_history, y_history를 plot
    plt.plot(x_history, y_history, 'o-')

    plt.show()

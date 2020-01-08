"""
파라미터 최적화 알고리즘 2) Momentum
1. Momentum 개념 설명
뉴턴 역학에서 물체의 질량과 속도의 곱으로 나타내는 물리량
 ** 운동량 p = 질량 m * 속도 v **
- 빨리 날라오는 공은 맞을 때 더 아프다
- 무거운 쇠공을 맞을 때 더 아프다
- 속도가 빠르면, 질량이 많으면 이동시간 더 빠르다
- 손에서 공을 놓으면 처음에는 속도 0이지만 속도 점점 더 빨라진다.

2. 알고리즘 설명
v : 속도 (velocity)
m : 모멘텀 상수 (momentum constant)
lr : 학습률
W : 파라미터
v = m * v - lr * dL/dW
W = W + v = W + m * v - lr * dL/dW  (m * v 을 더해준 것뿐)
"""
import matplotlib.pyplot as plt
import numpy as np
from lab_dl.ch06.ex01_matplot3d import fn, fn_derivative

class Momentum:
    def __init__(self, lr=0.01, m=0.9):
        self.lr = lr  # 학습률
        self.m = m  # 모멘텀 상수 (속도 v에 곱해줄 상수)
        self.v = dict()  # 속도 - 왜 dict 가 있을까? x, y, z 축 등등의 축의 방향이 있기 때문에

    def update(self, params, gradients):
        if not self.v:  # dict 의 원소가 없으면 : {} false 취급 & not = True
            # 초기값 지정 - 속도 0 에서부터 출발
            for key in params:
                # 파라미터(W, b등)와 동일한 shape 의 0으로 채워진 배열 생성
                self.v[key] = np.zeros_like(params[key])
        else:
            # 속도 v, 파라미터 params 를 갱신(update)하는 기능
            for key in params:
                # v = m * v - lr * dL/dW
                # self.v[key] = self.m * self.v[key] - self.lr * gradients[key]
                self.v[key] *= self.m
                self.v[key] -= self.lr * gradients[key]
                # W = W + v
                params[key] += self.v[key]


if __name__ == '__main__':
    # momentum 클래스 인스턴스 생성 / params, gradients 초기값 생성
    momentum = Momentum(lr=0.08)
    params = {'x': -7.0, 'y': 2.0}  # 실수로 바꿔주어야 update 실행 가능 (line 40)
    gradients = {'x': 0, 'y': 0}
    
    # 파라미터 갱신 값들을 저장할 리스트
    x_history = []
    y_history = []

    # update
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        momentum.update(params, gradients)

    # 점의 이동 방향 보기
    for x, y in zip(x_history, y_history):
        print(f'{x}, {y}')

    """
    y 의 절댓값을 보면, 증가했다 감소하는 경향이 있다.
    """

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
    plt.title('Momentum')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()

    """
    momentum 해석
    움직이는 모습이 훨씬 부드럽다.
    그릇을 왓다갔다 하면서 움직이듯이 
    """
    
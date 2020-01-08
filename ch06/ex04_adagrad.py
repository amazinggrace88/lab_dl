"""
파라미터 최적화 알고리즘
3) AdaGrad (Adaptive Gradient)
basic - SGD : W = W - lr * grad 에서는 학습률이 고정되어 있음
학습률을 계속 자동으로 바꿔주면서 gradient 를 찾으면 어떨까? -> AdaGrad 의 출발점

학습률을 변화시키면서 파라미터를 최적화함
** 학습률 변화시키는 방법 ** 
h 변수 도입
: 처음에는 큰 학습률로 시작, 반복할 때마다 학습률을 줄여나가면서 파라미터를 갱신한다.

h = h + grad * grad - grad 제곱 더해주므로 계속 커진다.
new_lr = lr / sqrt(h) - 반복할 때마다 lr 가 점점 더 작아진다.
W = W - (lr/sqrt(h) - new_lr) * grad

단점
반복할 때마다 h가 커지므로
분모가 커질수록 (lr/sqrt(h) - new_lr) 이 0이 되므로 파라미터 W 가 갱신이 약해진다.
"""
import matplotlib.pyplot as plt
import numpy as np
from lab_dl.ch06.ex01_matplot3d import fn, fn_derivative

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = dict()  # 새로운 h를 만드는 데 grad ** 2 를 더하므로 h 도 grad 와 모양이 같아야 한다.

    def update(self, params, gradients):
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])
        else:
            for key in params:
                self.h[key] += gradients[key] * gradients[key]  # element 별 제곱 (dot 이 아니다!)
                epsilon = 1e-8  # 0으로 나누는 것을 방지하기 위해서
                params[key] -= (self.lr / np.sqrt(self.h[key] + epsilon)) * gradients[key]


if __name__ == '__main__':
    # adagrad 클래스 인스턴스 생성 / params, gradients 초기값 생성
    adagrad = AdaGrad(lr=1.5)  # 반복될수록 작아지므로 lr 의 초기값이 작으면 의미가 없다.
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
        adagrad.update(params, gradients)

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
    plt.title('AdaGrad')

    plt.plot(x_history, y_history, 'o-', color='red')
    plt.show()


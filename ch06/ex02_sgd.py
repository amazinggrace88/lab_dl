"""
SGD - Stochastic Gradient Descent

<ch06>
신경망 학습 목적 : 손실 함수의 값을 가능한 낮추는 파라미터(W, b)를 찾는 것
                x -> () -> () -> Loss 사이에 끼어지는 W, b 찾기 위함
파라미터(parameter) :
    - 좁은 의미의 파라미터 : 가중치(Weight), 편향(bias)
    - 넓은 의미의 파라미터 : 하이퍼 파라미터(hyper parameter) - learning_rate(학습률), epochs(에포크), batch_size(배치 사이즈)
                                                         신경망에서의 hidden_layer 갯수(은닉층 갯수)
re 신경망의 학습 목적: (ch06 의 목표)
    - 좁은 의미의 파라미터를 갱신 (SGD, Momentum, AdoGrad, Adam)
    - 넓은 의미의 파라미터를 최적화 ()
"""
# SGD - Stochastic Gradient Descent 확률적 경사하강법
from lab_dl.ch06.ex01_matplot3d import fn_derivative, fn
import numpy as np
import matplotlib.pyplot as plt

class Sgd:
    """
    W = W - lr * dL/dW
    W : 파라미터 (가중치 or 편향 모두 가능)
    lr : learning_rate 학습률
    dL/dW : gradient 변화율 - 미분 연쇄 법칙을 적용 (오차역전파 이용)
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """
        파라미터 params 와 변화율 gradients 가 주어지면 파라미터를 갱신하는 메소드
        params, gradients : 딕셔너리라고 가정 - why? self.params['W1'] 이 dict 라서 (key, value 가 있다)
        """
        for key in params:
            # W = W - lr * dL/dW
            params[key] -= self.learning_rate * gradients[key]


if __name__ == '__main__':
    # Sgd 클래스의 객체 object ~ (인스턴스 instance) 생성 : 객체와 인스턴스는 살짝 다르다.
    sgd = Sgd(learning_rate=0.9)
    # ex01 module 에서 작성한 fn(x, y) 함수의 최솟값을 임의의 점에서 시작해서 원점을 찾아감.
    init_position = (-7, 2)
    # 신경망에서 찾고자 하는 파라미터의 초깃값
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]
    # 각 파라미터에 대한 변화율(gradient)
    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0

    # 각 파라미터 갱신 값들을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        sgd.update(params, gradients)

    for x, y in zip(x_history, y_history):
        print(f'({x}, {y})')

    # f(x, y) 함수를 등고선으로 표현
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    mask = Z > 7  # 바깥쪽으로 나올 수록 Z 가 커진다. 7보다 큰 Z는 모두 0이 되어 보이지 않는다.
    Z[mask] = 0

    plt.contour(X, Y, Z, 10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 등고선 그래프에 파라미터(x, y)들이 갱신되는 과정을 추가.
    plt.plot(x_history, y_history, 'o-', color='red')  # 점 o, 점과 점 사이 -
    plt.show()

    """
    learning_rate 0.95 : 2차원에서 y 좌표가 골짜기를 왔다갔다 하면서 움직이고 있다.
    단점 : 속도가 매우 느리다
    """


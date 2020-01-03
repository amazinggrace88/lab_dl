"""
경사하강법 (gradient descent)

**요약**
x_new = x_init - lr * df/dx
위 과정을 반복하면 f(x)의 최솟값을 찾음
- lr : 학습률
- 반복횟수 : epoch 에포크
"""
import numpy as np
import matplotlib.pyplot as plt
from lab_dl.ch04.ex05 import numerical_gradient


def gradient_method(fn, x_init, lr=0.01, step=100):
    x = x_init  # 점진적으로 변화시킬 변수
    x_history = []  # x가 변화되는 과정을 저장할 배열
    for i in range(step):
        x_history.append(x.copy())  # why? x 가 배열이기 때문에, 배열이 있는 곳의 주소값을 저장하게 됨.
        grad = numerical_gradient(fn, x)
        x -= lr * grad  # x_new = x_init - lr*grad 로 x 를 변경
    return x, np.array(x_history)

"""
x_history.append(x.copy()) 설명
1) x를 append 하면 배열에 주소값을 append
2) 만약 append 하는 x 의 주소값이 바뀐다면, 이전에 남아있는 append된 값은 주소가 없어 출력이 되지 않는다.
"""

def fn(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


if __name__ == '__main__':
    init_x = np.array([4.0])
    x, x_hist = gradient_method(fn, init_x, lr=0.1)  # lr 도 for 문으로 돌리면 된다.
    print('x = ', x)
    print('x_hist = \n', x_hist)


    # init_x 바꾸기
    init_x = np.array([4.0, -3.0])
    x, x_hist = gradient_method(fn, init_x, lr=0.1, step=100)
    print('x = ', x)
    print('x_hist = \n', x_hist)

    # x_hist 산점도 그래프 - 학습률(lr)이 너무 크면 최솟값을 찾지 못하고 발산, 학습률(lr)이 너무 작으면 속도 매우 느려짐
    plt.scatter(x_hist[:, 0], x_hist[:, 1])
    # 동심원
    # x**2 + y**2 = r**2 -> y**2 = r**2 - x**2
    for r in range(1, 5):
        r = float(r)  # 정수 -> 실수 변환
        x_pts = np.linspace(-r, r, 100)
        y_pts1 = np.sqrt(r**2 - x_pts**2)  # 위쪽 반원
        y_pts2 = -np.sqrt(r**2 - x_pts**2)  # 아래쪽 반원
        plt.plot(x_pts, y_pts1, ':', color='gray')
        plt.plot(x_pts, y_pts2, ':', color='gray')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.axvline(color='0.8')
    plt.axhline(color='0.8')
    plt.show()


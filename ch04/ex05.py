"""
4.3 수치미분
"""
import numpy as np


def numerical_diff(fn, x):
    """
    Numerical Differential 
    함수 fn, 점 x 에서의 기울기 (gradient) 리턴
    x 에서의 함수의 미분값
    """
    h = 1e-4  # 0.0001 너무 작게 잡아도 underflow 가 발생 가능
    return (fn(x + h) - fn(x - h)) / (2*h)


def f1(x):
    return 0.001 * x**2 + 0.01 * x


def f1_prime(x):  # f1 의 도함수 직접 구함
    return 0.002 * x + 0.01


"""
편미분
f(x_0, x_1) = x_0**2 + x_1**2 
한 가지 변수에 대해서만 미분 (다른 변수 상수 취급)
두 변수 모두에 대해서 한꺼번에 미분 불가능하다
"""


def f2(x):
    """x = [x0, x1] 이지만 원소 몇개가 들어와도 상관없음, [x0, x1, ... ]"""
    return np.sum(x**2)


def f3(x):
    """x = [x0, x1, x2]"""
    return x[0] + x[1] ** 2 + x[2] ** 3


def f4(x):
    """x = [x0, x1]"""
    return x[0] ** 2 + x[0] * x[1] + x[1] ** 2


def _numerical_gradient(fn, x):
    """
    점 x 에서의 함수의 각 편미분(partial differential)들의 배열
    fn : fn(x0, x1, x2..)
    x : [x0, x1, ...] => n차원
    """
    x = x.astype(np.float64, copy=False)  # 무조건 실수타입으로 넣는 것을 가정하자.
    gradient = np.zeros_like(x)  # = np.zeros(shape=x.shape) 원소 2개 짜리 array
    h = 1e-4  # 0.0001
    for i in range(x.size):  # 모든 원소의 갯수만큼 : x.size
        ith_value = x[i]
        # fn(x_i + h)
        x[i] = ith_value + h
        fh1 = fn(x)
        # fn(x_i - h)
        x[i] = ith_value - h
        fh2 = fn(x)
        gradient[i] = (fh1 - fh2) / (2*h)
        x[i] = ith_value  # x[i]번째를 원래 숫자로 되돌려놔야됨
    return gradient


def numerical_gradient(fn, x):
    """2 차원 용 numerical gradient !
    x = [
    [x11, x12, .. ]
    [x21, x22. .. ]
    ...            ]"""
    # print('in numerical gradient')
    # print('x = ', x)
    if x.ndim == 1:
        return _numerical_gradient(fn, x)
    else:
        grads = np.zeros_like(x)
        for i, x_i in enumerate(x):  # enumerate i(인덱스 출력), x_i(인덱스에 있는 데이터 출력)
            grads[i] = _numerical_gradient(fn, x_i)
            # print(f'grads[{i}] = {grads[i]}')
        return grads


if __name__ == '__main__':
    # 실제값 과 수치미분 값 얼마나 다른지 비교해보자
    # 실제값 f(x) = 0.001x^2 + 0.01x -> f'(x) = 0.002x + 0.01 -> f'(3) = 0.016

    print('근사값 predict differential : \n', numerical_diff(f1, 3))  # 0.016000000000043757
    print('실제값 true differential : \n', f1_prime(3))  # 0.016

    # f2 함수의 점(3, 4)에서의 편미분 값
    # f2 의 원소 1개를 numerical differential 에 넣어버리자.
    estimate_1 = numerical_diff(lambda x: x**2 + 4**2, 3)  # x1 을 상수로 넣었다(4)
    print('편미분1_x0을 변수, x1 상수: ', estimate_1)  # 6.0
    estimate_2 = numerical_diff(lambda x: 3**2 + x**2, 4)  # x0 을 상수로 넣었다(3)
    print('편미분2_x1을 변수, x0 상수: ', estimate_2)  # 7.9

    # 도함수들의 배열 만들기
    gradient = _numerical_gradient(f2, np.array([3.0, 4.0]))
    print('gradient array result : ', gradient)

    # 실습
    # f3 = x0 + x1**2 + x2**3
    # 점 (1, 1, 1)에서의 각 편미분들의 값
    # df/dx0 = 1, df/dx1 = 2, df/dx2 = 3
    gradient_f3 = _numerical_gradient(f3, np.array([1, 1, 1]))
    print('gradient f3 array result : ', gradient_f3)

    # f4 = x0**2 + x0 * x1 + x1**2
    # 점 (1, 2)에서의 df/dx0 = 4, df/dx1 = 5
    gradient_f4 = _numerical_gradient(f4, np.array([1, 2]))
    print('gradient f4 array result : ', gradient_f4)

    # 복습
    # gradient -> f(weight, bias) = Entropy 라는 함수의 기울기와 같다.
    # 1) Entropy 의 점을 랜덤으로 잡는다 = weight 초기값(random 설정)
    # 2) gradient 를 계산하여 음수 ->  weight 양의 방향으로 움직임, 양수 -> weight 음의 방향으로 움직이면 최솟값을 찾게 된다.

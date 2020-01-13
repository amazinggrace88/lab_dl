"""
1 차원 convolution 이라는 건 사실 Cross-correlation (교차상관) 연산
cf. 1 차원 convolution (합성곱) 은 관습적으로 사용하는 표현
cf. 합성곱과 교차상관은 매우 비슷하다. (대칭적)
- 교차상관이라는 메소드를 사용하면서 합성곱이라고 부르고 있음
"""
import numpy as np


def convolution_1d(x, w):
    """
    # x와 w의 Convolution (합성곱 연산) 결과를 리턴
    x, w : 1d ndarray
    len(x) >= len(w)

    1) W 을 반전 (1차원 - 좌우 반전, 2차원 - 좌우 반전, 위아래 반전)
    2) FMA(Fused Multiply-Add) : 한칸씩 이동하면서 원소별로 곱하고 다 더해준다.
    - 1차원 convoluation 연산의 크기(원소의 갯수)는
        = len(x) - len(w) + 1
        ex) 5 - 1 + 1 = 5
        ex) 5 - 2 + 1 = 4
        ex) 5 - 3 + 1 = 3
    """
    w_r = np.flip(w)  # w 반전
    conv = cross_correlation_1d(x, w_r)
    return conv


def cross_correlation_1d(x, w):
    """
    x, w : 1d ndarray
    len(x) >= len(w)
    x 와 w 의 교차상관 연산 결과 리턴
    convolution_1d() 함수가 cross_correlation_1d() 를 사용하도록 수정

    <교차상관(Cross-Correlation) 연산>
    합성곱 연산과 다른 점 : w 를 반전시키지 않는다
    교차상관과 합성곱이 차이가 나지 않는 이유 - w라는 행렬을 난수로 만들어내어 forward, backward 하기 때문에
                                        반전시킨 것을 난수로 사용한다고 이해할 수 있다.
    CNN(Convolutional Neural Network, 합성곱 신경망)에서는 대부분의 경우 교차상관을 사용함
    """
    nx = len(x)  # 지정
    nw = len(w)
    n = nx - nw + 1
    conv = []
    if nx >= nw:
        for i in range(n):
            x_sub = x[i:i + nw]
            fma = np.sum(x_sub * w)
            conv.append(fma)
        conv = np.array(conv)
        return conv
    else:
        print('합성곱을 할 수 없습니다')


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 6)
    print('x = ', x)
    w = np.array([2, 1])
    print('w = ', w)

    # Convolution (합성곱 연산)
    """
    1) W 을 반전 (1차원 - 좌우 반전, 2차원 - 좌우 반전, 위아래 반전)
    2) FMA(Fused Multiply-Add) : 한칸씩 이동하면서 원소별로 곱하고 다 더해준다.
    - 1차원 convoluation 연산의 크기(원소의 갯수)는 
        = len(x) - len(w) + 1
        ex) 5 - 1 + 1 = 5
        ex) 5 - 2 + 1 = 4
        ex) 5 - 3 + 1 = 3
    """
    # x Conv w (합성곱) - w 가 x 보다 작거나 같아야 한다.
    if len(x) >= len(w):
        w_r = np.flip(w)
        print('w_r = ', w_r)
        conv = []
        for i in range(len(x) - len(w_r) + 1):
            x_sub = x[i:i+2]  # x에서 원소 2개 꺼냄 (0, 1), (1, 2), (2, 3), (3, 4)
            fma = x_sub.dot(w_r)
            # fma = np.sum(x_sub * w_r)와 같은 의미
            conv.append(fma)
        conv = np.array(conv)
        print('conv = ', conv)
    else:
        print('합성곱을 할 수 없습니다')

    # def 실험
    result = convolution_1d(x, w)
    print('convolution function = ', result)
    x = np.arange(1, 6)
    w = np.array([2, 0, 1])
    result = convolution_1d(x, w)
    print('convolution function = ', result)


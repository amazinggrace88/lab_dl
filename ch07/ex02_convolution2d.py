"""
2 차원 convolution
"""
import numpy as np


def convolution_2d(x, w):
    """
    교차상관
    : w 를 반전시키는 과정(flipping) 없음.
    x, w: 2d ndarray, x.shape >= w.shape
    x, w의 교차상관을 연산하자 (w 반전 시키지 않아도 됨)
    """
    rows = x.shape[0] - w.shape[0] + 1
    cols = x.shape[1] - w.shape[1] + 1
    conv = []
    for i in range(rows):
        for j in range(cols):
            x_sub = x[i:(i + w.shape[0]), j:(j + w.shape[1])]
            # print(f'x_sub = x[{i}:({i} + {w.shape[0]}), {j}:({j} + {w.shape[1]})]')
            fma = np.sum(x_sub * w)
            conv.append(fma)
    conv = np.array(conv)
    return conv


if __name__ == '__main__':
    np.random.seed(113)
    x = np.arange(1, 10).reshape((3, 3))
    print('x = \n', x)
    w = np.array([[2, 0],
                  [0, 0]])
    print('w = \n', w)

    # 2d 배열 x 의 가로(width) xw, 세로(hight) xh
    xh, xw = x.shape[0], x.shape[1]
    # 2d 배열 w 의 가로(width) ww, 세로(hight) wh
    wh, ww = w.shape[0], w.shape[1]

    # w 의 크기에 맞추어 x 의 부분집합 출력
    x_sub1 = x[0:wh, 0:ww]
    x_sub2 = x[0:wh, 1:ww+1]
    x_sub3 = x[1:wh+1, 0:ww]
    x_sub4 = x[1:wh+1, 1:ww+1]
    print('x_sub 1 = \n', x_sub1)
    print('x_sub 2 = \n', x_sub2)
    print('x_sub 3 = \n', x_sub3)
    print('x_sub 4 = \n', x_sub4)
    fma1 = np.sum(x_sub1 * w)
    fma2 = np.sum(x_sub2 * w)
    fma3 = np.sum(x_sub3 * w)
    fma4 = np.sum(x_sub4 * w)
    print('fma1 = ', fma1)
    print('fma2 = ', fma2)
    print('fma3 = ', fma3)
    print('fma4 = ', fma4)
    print('convolution function = \n', convolution_2d(x, w))

    x = np.random.randint(10, size=(5, 5))
    print('x = \n', x)  # 0~9 까지의 정수 중 난수
    w = np.random.randint(5, size=(3, 3))
    print('w = \n', w)
    result = convolution_2d(x, w)
    print('2d convolution = ', result)

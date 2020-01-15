"""
Convolution 클래스를 만들기
"""
from lab_dl.lab_dl.common.util import im2col
from lab_dl.lab_dl.dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 가중치 행렬 - filter 역할
        self.b = b
        self.stride = stride
        self.pad = pad
        # 중간 데이터 : forwad 생성되는 데이터 -> backward 에서 사용
        self.X = None
        self.x_col = None
        self.W_col = None
        # gradient 들 - backward 에서 저장되는 값들
        self.dW = None
        self.db = None

    def forward(self, x):
        # x : 4차원 이미지 (mini-batch) 데이터
        self.x = x
        n, c, h, w = self.x.shape
        fn, c, fh, fw = self.W.shape
        oh = ((h - fh + self.pad*2) // self.stride) + 1
        ow = ((w - fw + self.pad*2) // self.stride) + 1

        self.x_col = im2col(x, fh, fw, self.stride, self.pad)
        # w_col = self.W.reshape(fn, c*fh*fw).T - c*fh*fw의 의미 : 1차원으로 만든다.
        self.w_col = self.W.reshape(fn, -1).T
        out = self.x_col.dot(self.w_col) + self.b  # np.메소드(x, w) = x.npmethod(w) : x 가 numpy 배열이어야 함
        out = out.reshape(n, oh, ow, fn).transpose(0, 3, 1, 2)  # -1: fn으로 대체 가능
        return out


if __name__ == '__main__':
    np.random.seed(115)
    # W, d 생성
    # filter : (fn, c, fn, fw) = (1, 2, 4, 4)
    W = np.zeros((1, 1, 4, 4), dtype=np.uint8)  # uint8: 부호없는 정수
    W[0, 0, 1, :] = 1  # 세로 필터
    # W[0, 0, :, 1] = 1  # 가로 필터
    print('W : \n', W)
    b = np.zeros(1)
    print('b : \n', b)

    # convolution 생성
    convolution = Convolution(W, b)

    # MNIST 데이터 forward
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)  # 4차원 형식 그대로 사용
    print('x_train : ', x_train.shape)

    # 3차원 - convolution.forward 는 4차원용 함수
    # slicing 을 활용하여 4차원으로 만들 수 있다.
    input = x_train[0]
    print('input :', input.shape)  # (1, 28, 28)
    input = x_train[0:1]
    print('input :', input.shape)  # (1, 1, 28, 28)
    output = convolution.forward(input)
    print('output : ', output.shape)  # (60000, 1, 25, 25)

    # graph
    img = output.squeeze()
    print('img :', img.shape)  # 차원을 하나씩 없애주는 것
    plt.imshow(img, cmap='gray')
    plt.show()

    # 다운로드 받을 이미지파일 ndarray 로 변환해서 forward

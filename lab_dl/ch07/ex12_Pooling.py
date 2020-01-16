from lab_dl.lab_dl.common.util import im2col
import numpy as np
import matplotlib.pyplot as plt

from lab_dl.lab_dl.dataset.mnist import load_mnist


class Pooling:
    def __init__(self, fh, fw, stride=1, pad=0):
        self.fh = fh  # pooling 윈도우의 높이(height)
        self.fw = fw  # pooling 윈도우의 너비(width)
        self.stride = stride  # pooling 윈도우를 이동시키는 크기(보폭)
        self.pad = pad  # 패딩 크기
        # backward에서 사용하게 될 값
        self.x = None  # pooling 레이어로 forward되는 데이터 - backward 을 위해 저장함
        self.arg_max = None  # 찾은 최댓값의 인덱스

    def forward(self, x):
        """x: (samples, channel, height, width) 모양의 4차원 배열"""
        # 구현
        self.x = x
        n, c, h, w = self.x.shape
        out_h = int(1 + (h - self.fh + 2*self.pad) / self.stride)
        out_w = int(1 + (h - self.fw + 2*self.pad) / self.stride)
        # 1) x --> im2col --> 2차원 변환
        col = im2col(self.x, self.fh, self.fw, self.stride, self.pad)  # col.shape : (n*oh*ow, c*fh*fw)
        # 2) 채널 별 최댓값을 찾을 수 있는 모양으로 x를 reshape
        col = col.reshape(-1, self.fh * self.fw)
        # 3) 채널 별로 최댓값을 찾음.
        self.arg_max = np.argmax(col, axis=1)  # 최대값의 위치 - backward 을 위해 저장함
        out = np.max(col, axis=1)
        # 4) 최댓값(1차원 배열)을 reshape & transpose : (n, oh, ow, c)
        out = out.reshape(n, out_h, out_w, c)
        # 5) pooling이 끝난 4차원 배열을 리턴
        out = out.transpose(0, 3, 1, 2)
        return out


if __name__ == '__main__':
    np.random.seed(116)
    # pooling 클래스의 forward 메소드를 테스트
    # x 를 (1, 3, 4, 4) 모양으로 무작위로(랜덤하게) 생성, 테스트
    x = np.random.randint(10, size=(1, 3, 4, 4))
    print('x : ', x)
    
    # Pooling 클래스의 인스턴스 생성
    pool = Pooling(fh=2, fw=2, stride=2, pad=0)
    out = pool.forward(x)
    print('out : ', out)

    # MNIST 데이터를 로드
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=False, flatten=False)

    x = x_train[:5]
    print('x shape : ', x.shape)
    out = pool.forward(x)
    print('out shape : ', out.shape)
    # 학습 데이터 중에서 5개를 mini-batch 로 forward
    for i in range(5):
        # ax = plt.subplot(row, col, (i + 1))
        # row, col 로 공간을 나누고 plot 각각 삽입
        # 인덱스 1번부터 시작 : 1번 영역 = (0+1) / 2번 영역 = (1+1) / ..
        # xticks=[], yticks=[] : tick(축 표시) 를 비움
        ax = plt.subplot(2, 5, (i + 1), xticks=[], yticks=[])
        plt.imshow(x[i].squeeze(), cmap='gray')  # 학습데이터 채널 축을 없애야 (squeeze()) - matplotlib 에서 동작 가능
        ax2 = plt.subplot(2, 5, (i + 6), xticks=[], yticks=[])
        plt.imshow(out[i].squeeze(), cmap='gray')
    plt.show()






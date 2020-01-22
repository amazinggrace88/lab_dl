"""
CNN 이 사용하는 파라미터(filter W, bias b)의 초깃값과 학습 후 W, b 값 비교
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

from lab_dl.lab_dl.ch07.simple_convnet import SimpleConvNet
from lab_dl.lab_dl.common.layers import Convolution


def show_filter(filters, num_filters, ncols=8):
    # subplot 사용
    # ncols = 컬럼갯수
    # num_filters // ncols = nrows
    # nrows * ncols 바둑모양으로 구분되어 그래프 생성하기
    nrows = np.ceil(num_filters / ncols)  # ceil 천장 -> 30 / 8 = 3.75‬ 이면 4 출력
    for i in range(num_filters):
        # subplot 위치 결정
        plt.subplot(nrows, ncols, (i+1), xticks=[], yticks=[])
        # 그래프 넣기
        plt.imshow(filters[i, 0], cmap='gray')
        # filters[i, 0] 의 의미 : 4차원에서 2차원을 꺼낸다
        # 4차원에서 인덱스 0번-> 3차원 의 인덱스 0번 -> 2차원 배열 (5, 5)
    plt.show()


if __name__ == '__main__':
    # Simple CNN 생성
    cnn = SimpleConvNet()
    # 학습시키기 전 파라미터 - 난수들로 이루어짐
    before_filters = cnn.params['W1']
    print(before_filters.shape)
    # (30, 1, 5, 5) -> filter 갯수 30개, (1, 5, 5) 이므로
    # filter = 30개
    show_filter(before_filters, 30, ncols=8)  # 0 검은색, 255 흰색으로 변환된 숫자가 나옴..

    # 학습 시킨 후 파라미터 - pkl 에 있음
    # pickle 파일에 저장된 파라미터를 cnn의 필드로 로드.
    cnn.load_params('cnn_params.pkl')

    after_filters = cnn.params['W1']
    # 학습 끝난 후 갱신(업데이트)된 파라미터를 그래프로 출력
    show_filter(after_filters, 16, ncols=4)

    # 학습 끝난 후 갱신된 파라미터를 실제 이미지 파일에 적용
    # pyplot.imread : png 파일을 np.array 로 변환함
    # jpeg 파일은 PIL.library 를 통해 np.array 로 변환해야함
    lena = imread('lena_gray.png')
    print(lena.shape)

    # 이미지 데이터를 Convolution 레이어의 forward() 메소드에 전달.
    # lena - 2차원 <-> 레이어는 무조건 4차원
    # -> 2차원 4차원으로 변환하는 작업 거침
    # *lena.shape : () tuple 에서 숫자 1개씩 뽑아준다
    lena = lena.reshape(1, 1, *lena.shape)
    for i in range(16):
        # 필터
        w = cnn.params['W1'][i]  # 갱신된 필터
        # b = cnn.params['b1'][i]  # 갱신된 바이어스
        b = 0
        w = w.reshape(1, *w.shape)  # 3차원 -> 4차원으로 변환
        conv = Convolution(w, b)  # Convolution 레이어 생성
        out = conv.forward(lena)  # i 번 필터를 forward 하여 지나가게 함.
        # pyplot 을 사용하기 위해서 4차원을 2차원으로 변환
        out = out.reshape(out.shape[2], out.shape[3])
        plt.subplot(4, 4, i+1, xticks=[], yticks=[])
        plt.imshow(out, cmap='gray')
    plt.show()




"""
convolution (합성곱) padding 적용한 함수 사용
"""
import numpy as np
from scipy.signal import convolve, correlate, convolve2d, correlate2d

from lab_dl.ch07.ex01_convolution1d import convolution_1d

if __name__ == '__main__':
    x = np.arange(1, 6)
    w = np.array([2, 0])
    print('===== convolution 비교 =====')
    print('before = ', convolution_1d(x, w))
    """
    일반적인 convolution(x, w) 결과는 (4, ) 일 것!
    convolution 연산에서 x 의 원소 중 1, 5 는 연산에 1번만 기여. 다른 원소들은 2번씩 기여함 -> padding 필요한 이유
    """
    # 1. padding 함수 사용
    # x 의 모든 원소가 convolution 연산에서 동일한 기여를 할 수 있도록 padding 하라
    # 앞 뒤 padding 의 갯수 = w 의 원소의 갯수(가중치의 크기) - 1
    # 결과는 x 의 모양 보다 더 커진다!
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print('after = ', convolution_1d(x_pad, w))

    # convolution 결과의 크기가 입력 데이터 x 와 동일한 크기가 되도록 padding
    # padding 1 군데에만 넣으면 padding 갯수 최소화하면서도 모든 원소들이 2번씩 기여
    # before-padding 에만 정책을 정하면
    x_pad = np.pad(x, pad_width=(1, 0), mode='constant', constant_values=0)
    print('before padding = 1 인 경우 = ', convolution_1d(x_pad, w))
    # after-padding 에만 정책을 정하면
    x_pad = np.pad(x, pad_width=(0, 1), mode='constant', constant_values=0)
    print('after padding = 1 인 경우 = ', convolution_1d(x_pad, w))

    # 2. convolution 함수 사용
    # 함수 사용 : scipy.signal.convolve - mode 를 변경하면서 padding 까지 포함한 concolution 함수를 반환함
    conv = convolve(x, w, mode='valid')  # padding 없음
    print('convolve 함수 사용 / valid mode / no padding = ', conv)
    conv = convolve(x, w, mode='full')
    print('convolve 함수 사용 / full mode (before, after padding) = ', conv)
    conv = convolve(x, w, mode='same')  # before padding
    print('convolve 함수 사용 / same mode / before padding = ', conv)
    # 함수 사용 : scipy.signal.correlate - 1 차원에서만! mode 를 변경하면서 padding 까지 포함한 convolution 함수를 반환함
    cross_corr = correlate(x, w, mode='valid')
    print('cross_corr / no padding = ', cross_corr)
    cross_corr = correlate(x, w, mode='full')
    print('cross_corr / before, after padding = ', cross_corr)
    cross_corr = correlate(x, w, mode='same')
    print('cross_corr / before padding = ', cross_corr)

    # 5. 2 차원 convolution 함수 사용
    # 함수 사용 : scipy.signal.convolve2d, scipy.signal.correlate2d()
    # (4, 4) 2d ndarray
    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    # (3, 3) 2d ndarray
    w = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])
    # x, w 의 교차 상관 연산
    # valid = (2, 2)
    cross_corr = correlate2d(x, w, mode='valid')  # no padding
    print('cross corr shape / valid = ', cross_corr.shape)
    print('cross corr shape / valid matrix - no padding = ', cross_corr, sep='\n')
    # full = (6, 6)
    cross_corr = correlate2d(x, w, mode='full')  # before, after padding
    print('cross corr shape / full = ', cross_corr.shape)
    print('cross corr shape / full matrix = ', cross_corr, sep='\n')
    # same = (4, 4)
    cross_corr = correlate2d(x, w, mode='same')  # before padding - x에 맞춰서 shape 이 나옴
    print('cross corr shape / same = ', cross_corr.shape)
    print('cross corr shape / same(before padding) matrix = ', cross_corr, sep='\n')

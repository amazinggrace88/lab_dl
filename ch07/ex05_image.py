"""
CNN(Convolutional Neural Network, 합성곱 신경망)
원래 convolution 연산은 영상/음성 처리(image/audio processing)에서
신호를 변환하기 위한 연산으로 사용.
"""
import numpy as np
from PIL import Image  # 신호 - signal 이라고 번역되어 scipy.signal 이라는 모듈 구성
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate  # 3 차원 그대로 사용 가능한 함수

# jpeg 파일 오픈
img = Image.open('dubai.jpg')
img_pixel = np.array(img)
print(img_pixel.shape)
"""
image 의 모양
shape (1280, 1920, 3) 
= 1920 width * 1280 height jpeg (24-bit color)
= 행렬이기 때문에 height, width, layer 3장(R,G,B) = (row, column, color-depth) 으로 출력

* 주의 ! 
color-depth 는 머신러닝 라이브러리에 따라 color 표기의 위치가 가장 처음이거나 나중에 나올 수 있다 (달라진다)
Tensorflow : channel-last 방식, color-depth 가 n-D 배열의 마지막 차원이다
Theano : channel-first 방식, color-depth 가 n-D 배열의 첫번째 차원이다
keras : 두가지 방식 모두 지원
"""
# pyplot 으로 이미지 보기
plt.imshow(img_pixel)
plt.show()

# 이미지의 RED 값 정보
"""
배열에 대한 정보
# 2 차원 배열은 1차원 배열을 원소로 갖는 배열
# 3 차원 배열은 2차원 배열을 원소로 갖는 배열
- 2차원 배열을 층층히 쌓아올린 것 = 3차원
- R, G, B layer 를 층층히 쌓아올린 것이 사진이 된다.
(cf. 동영상 + 시간 축)
"""
print('Red 축 =', img_pixel[:, :, 0], sep='\n')
print('Green 축 =', img_pixel[:, :, 1], sep='\n')
print('Blue 축 =', img_pixel[:, :, 2], sep='\n')

# (3, 3, 3) 필터로 filtering
filter = np.zeros((3, 3, 3))  # convolution 하면 모두 0 이 되는 filter - 마지막 차원의 컬러의 크기를 원본 컬러에 맞춰야함
filter[1, 1, 0] = 1.0
transformed = convolve(img_pixel, filter, mode='same') / 255
plt.imshow(transformed)
plt.show()
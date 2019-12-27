"""
MNIST 숫자 손글씨 데이터 세트
"""
from PIL import Image
from dataset.mnist import load_mnist
import numpy as np


def img_show(img_arr):
    """Numpy 배열(ndarray)로 작성된 이미지를 화면 출력"""
    img = Image.fromarray(np.uint8(img_arr))  # numpy 배열 형식을 이미지 객체로 변환(array 에서 image 만들어준다)
    # uint8() : 8 bits 를 부호가 없는 숫자로 즉, 0~256(2^8)-1 범위로 주어라
    #           8 bits 를 부호를 가지는 숫자면 -128~127(-2^7~2^7-1) 범위가 된다. 이를 피하기 위함
    img.show()


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=False)
    # mnist.pkl 이 자동으로 만들어짐, 첫번째로는 파일을 다운받는 것이고, 다음 부터는 있는 pickle 파일에 접속
    # (학습 이미지 60,000 장 데이터, 학습 데이터 레이블), (테스트 이미지 데이터 세트, 테스트 데이터 레이블) 두개의 튜플
    print('X train shape : ', X_train.shape)
    # (60000, 784) : 28 * 28(=784) 크기의 이미지 60,000장
    print('y train shape : ', y_train.shape)
    # (60000, ) : 60,000개 손글씨 이미지 숫자들(레이블)

    # 학습 세트의 첫번째 이미지
    img = X_train[0]  # 첫번째 행
    img = img.reshape((28, 28))  # 이미지는 1차원 리스트 모양이 아니라 28 * 28 행렬로 변환해야 볼 수 있다. - 전체 점의 갯수
    print(img)
    # 이미지는 점(pixel)로 이루어지는데, 점에 색칠을 하면(255-8bit -1) 하얗게, 색칠을 안하면(0) 검게 된다.
    # 숫자들의 배열로 그림을 나타낼 수 있다 !
    # 이미지를 3개 가지고 있고 RGB의 색 줄 수 있다 (겹치게 보이게 함)
    # 판이 3개면 색감있는 그림을 숫자로 표현 가능함
    img_show(img)  # 2차원 넘파일 배열을 이미지로 출력
    print('label : ', y_train[0])

    
    # feature control
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
    
    print('X_train shape : ', X_train.shape)
    # (60000, 1흑백, 28, 28)  : flatten=False 이기 때문에 784 -> 28 * 28 로 변환하였다. (형태 reshape 으로 바꿀 필요가 없다)
    # flatten = False 인 경우 이미지 구성을 (컬러, 가로, 세로) 형식으로 표시함
    # (신경망 넣을때는 입력값으로 펼쳐놓고 flatten true 로 들어가는게 낫다)
    
    print('y_train shape : ', y_train.shape)
    # (60000, 10) : one_hot_label=True 이기 때문에 행과 열을 둘 다 가짐.
    # one_hot_encoding 을 지원하는 경우이기 때문에 .. 원소 10개짜리 리스트가 열 갯수가 된다.
    print('y_train[0]', y_train[0])
    # one_hot_encoding ?
    # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 0~9 숫자 중 5에만 1 (hot) - 0~9 가능성 중 5 에만 1 표시
    # 핫태핫태~ 나중에 신경망의 출력층도 이런 식으로 만든다.

    img = X_train[0]
    print(img)
    # normalize=True인 경우, 각 픽셀의 숫자들이 0~1 사이의 숫자들로 정규화됨 (신경망 넣을때 normalize true가 좋다)
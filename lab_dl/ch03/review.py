"""
신경망을 행렬로 나타내기
"""
import pickle
import numpy as np
from PIL import Image

from ch03.ex01 import sigmoid
from dataset.mnist import load_mnist


def init_network():  # 파일에서 가져오지 않고 가중치 행렬을 임의로 만들었다
    """
    W1,W2,W3 matrix
    bias matrix
    :

    """
    np.random.seed(1224)
    network = dict()

    network['W1'] = np.random.random(size=(2, 3)).round(2)
    print('W1 : \n', network['W1'])
    network['W2'] = np.random.random(size=(2, 3)).round(2)
    print('W2 : \n', network['W2'])
    network['W3'] = np.random.random(size=(2, 3)).round(2)
    print('W3 : \n', network['W3'])

    network['b1'] = np.random.random(3).round(2)
    print('b1 : \n', network['b1'])
    network['b2'] = np.random.random(3).round(2)
    print('b2 : \n', network['b2'])
    network['b3'] = np.random.random(3).round(2)
    print('b3 : \n', network['b3'])

    return network


def forward(network, x_test):
    """
    순방향 전파 : 입력, 은닉층, 출력
    """
    # 가중치 행렬
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 은닉층에서 활성화 함수 : sigmoid 함수 -> 첫번째 은닉층 전파
    a1 = x_test.dot(W1) + b1
    z1 = sigmoid(a1)
    # 두번째 은닉층 전파
    a2 = x_test.dot(W2) + b2
    z2 = sigmoid(a2)
    # 출력층
    y = z2.dot(W3) + b3
    return y  # 출력층에서 전파하지 않았음 - softmax 함수 작성 필


def softmax(x):
    """
    분류 문제에서의 활성화함수
    <특징>
    - 리턴값 0~1사이의 값
    - 모든 리턴값의 총합 1
    """
    max_x = np.max(x)
    y = np.exp(x - max_x) / np.sum(np.exp(x-max_x))
    return y


def img_show(img_arr):
    img = Image.fromarray(np.uint8(img_arr))
    img.show()


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def image_to_pixel(image_file):
    img = Image.open(image_file, mode='r')
    print(type(img))
    pixels = np.array(img)
    print('pixel shape : ', pixels.shape)
    return pixels



if __name__ == '__main__':
    # step 1. load_mnist 함수로 MNIST 데이터셋 읽기
    (X_train, y_train), (X_test, y_test) = load_mnist()
    # 신경망 에서 사용할 가중치와 편향(bias) 생성
    network = init_network()
    # step 2. 예측
    # step 3. 정확도 계산
    # step 4. 시각화
    img = X_train[0]
    img = img.reshape((28, 28))

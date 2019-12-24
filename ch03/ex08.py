"""
MINIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle
import numpy as np
from lab_dl.ch03.ex01 import sigmoid
from lab_dl.ch03.ex05 import identity_function, softmax
from lab_dl.dataset.mnist import load_mnist
from PIL import Image


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어옴
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 확인
    return network


def predict(network, X_test):
    """
    - 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴

    예측을 하려면 test set(array들)을 주어야 함

    :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
    :param x: 입력값을 가지고 있는 test set 의 원소
    """
    y_pred = []
    for x_i in X_test:
        # 가중치 행렬:
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        # bias 행렬:
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        # 은닉층에서 활성화 함수: sigmoid 함수
        a1 = x_i.dot(W1) + b1
        z1 = sigmoid(a1)  # 첫번째 은닉층 전파
        z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파
        # 출력층: z2 @ W3 + b3 값을 그대로 출력
        y = z2.dot(W3) + b3
        y_pred_elements = softmax(y)  # 분류이므로 softmax() 함수 사용하였음~
        y_pred.append(y_pred_elements)
    return y_pred


def accuracy(y_test, y_pred):
    """
    - 테스트 데이터 레이블과 테스트 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴

    레이블, 예측값 - 숫자 리턴

    :param y_test: 테스트 레이블 리스트
    :param y_pred: 테스트 데이터 리스트
    """
    accuracy_cnt = 0
    for i in range(len(y_test)):
        p = np.argmax(y_pred[i])
        if p == y_test[i]:
            accuracy_cnt += 1
    return accuracy_cnt / len(y_test)


def image_show(img_arr):
    img = Image.fromarray(np.uint8(img_arr))  # numpy array 형식을 이미지 객체로 변환(array -> image)
    img.show()


if __name__ == '__main__':
    # 신경망 생성
    network = init_network()
    print('network is .. ', network)
    
    # step 1. load_mnist 함수로 MNIST 데이터셋 읽기
    # (학습 이미지 60,000 장 데이터, 학습 데이터 레이블), (테스트 이미지 데이터 세트, 테스트 데이터 레이블) 두개의 튜플 만들기
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    
    # step 2. 예측
    y_pred = predict(network, X_test)
    print('y_pred is .. ', y_pred)
    
    # step 3. 정확도 계산
    acc = accuracy(y_test, y_pred)
    print('accuracy is .. ', acc)

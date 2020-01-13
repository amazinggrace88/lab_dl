"""
MINIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle
import numpy as np
from PIL import Image

from lab_dl.ch03.ex01 import sigmoid
from lab_dl.ch03.ex05 import softmax
from lab_dl.dataset.mnist import load_mnist


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어옴
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 확인
    return network


def forword(network, x):
    """
    :param x: image 한 개 정보를 가지고 있는 배열 (784, ) image 1장 즉, size만 있고 1차원
    """
    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 첫번째 은닉층 전파(propagation)
    z1 = sigmoid(x.dot(W1) + b1)
    # 두번째 은닉층 전파(propagation)
    z2 = sigmoid(z1.dot(W2) + b2)
    # 출력층 전파
    y = softmax(z2.dot(W3) + b3)

    return y


def predict(network, X_test):
    prediction = []
    for x_i in X_test:
        # 각 이미지를 신경망에 전파(통과)시켜 확률을 계산하는 과정 반복
        y_hat = forword(network, x_i)
        # 가장 큰 확률의 index(예측값) 를 준다.
        y_pred = np.argmax(y_hat)
        prediction.append(y_pred)
    return np.array(prediction)
#
#
# def predict(network, X_test):
#     """
#     - 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
#     테스트 데이터의 예측값(배열)을 리턴
#
#     예측을 하려면 test set(array들)을 주어야 함
#
#     :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
#     :param X_test: 입력값을 가지고 있는 test set 배열
#     """
#     y_pred = []
#     for x_i in X_test:
#         # 가중치 행렬:
#         W1, W2, W3 = network['W1'], network['W2'], network['W3']
#         # bias 행렬:
#         b1, b2, b3 = network['b1'], network['b2'], network['b3']
#         # 은닉층에서 활성화 함수: sigmoid 함수
#         a1 = x_i.dot(W1) + b1
#         z1 = sigmoid(a1)  # 첫번째 은닉층 전파
#         z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파
#         # 출력층: z2 @ W3 + b3 값을 그대로 출력
#         y = z2.dot(W3) + b3
#         y_pred_elements = softmax(y)  # 분류이므로 softmax() 함수 사용하였음~
#         y_pred.append(y_pred_elements)
#     return y_pred  # 행렬의 곱셈은 되지만, 예측확률이 (10000, 10) 이 되지만 softmax 함수가 동작을 잘 못함..? ex11 mini-batch 참고


def accuracy(y_true, y_pred):
    """
    - 테스트 데이터 레이블과 테스트 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴

    레이블, 예측값 - 숫자 리턴

    :param y_test: 테스트 레이블 리스트
    :param y_pred: 테스트 데이터 리스트
    """
    accuracy_cnt = 0
    for i in range(len(y_true)):
        p = np.argmax(y_pred[i])
        if p == y_true[i]:
            accuracy_cnt += 1
    return accuracy_cnt / len(y_true)  # for 문보다 numpy 의 기능이 속도 훨씬 빠르다. --> for 문 이해 안됨.


def accuracy(y_true, y_pred):
    result = (y_true == y_pred)  # 10000개 짜리 np.array 두개를 비교 -> 원소별로 비교 -> bool 을 저장하고 있는 np.array
    print(result[:10])
    return np.mean(result)  # numpy 의 기능 중 하나 -> 평균을 true 1, false 0으로 생각하여 평균 계산


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)  # 같은 의미


def image_show(img_arr):
    img = Image.fromarray(np.uint8(img_arr))  # numpy array 형식을 이미지 객체로 변환(array -> image)
    img.show()


if __name__ == '__main__':

    # step 1. load_mnist 함수로 MNIST 데이터셋 읽기
    # (학습 이미지 60,000 장 데이터, 학습 데이터 레이블), (테스트 이미지 데이터 세트, 테스트 데이터 레이블) 두개의 튜플 만들기
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print('X_train : ', X_train[0])
    print('y_train : ', y_train[0])

    # 신경망 에서 사용할 가중치와 편향(bias) 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print(f'w1 : {W1.shape}, w2 : {W2.shape}, w3 : {W3.shape}')
    print(f'b1 : {b1.shape}, b2 : {b2.shape}, b3 : {b3.shape}')
    # w1: (784, 50), w3: (50, 100), w3: (100, 10)
    # w1: 입력층이 784개, 뉴런은 50개(첫번째 은닉층) -> 편향 50개(b1)
    # w2: 두번째 은닉층의 입력층은 50개, 뉴런은 100개(두번째 은닉층) -> 편향 100개(b2)
    # w3: 출력층의 입력층은 50개, 출력층은 10개 -> 편향 10개(b3)
    # 신경망은 설계 가능하다 - 즉, init_network 설계 가능하다
    
    # step 2. 예측
    y_pred = predict(network, X_test)
    print('예측값 : ', y_pred.shape)
    print('y_pred : ', y_pred[:10])
    print('y_test : ', y_test[:10])

    # step 3. 정확도 계산
    acc = accuracy(y_test, y_pred)
    print('accuracy : ', acc)

    # step 4. 시각화
    # 예측이 틀린 첫번째 이미지 : X_test[8]
    # 문제점 : noralize(0~1) 되어 있고, 1차원 배열로 flatten 된 데이터~!
    img = X_test[8] * 255  # 0~1 -> 0~255 사이의 값으로 바꿈 (역정규화)
    img = img.reshape((28, 28))  # 1차원 배열 -> 2차원 배열
    img = Image.fromarray(img)  # 2차원 numpy 배열을 이미지로 변환
    img.show()



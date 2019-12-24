"""
MINIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

from lab_dl.ch03.ex01 import sigmoid
from lab_dl.ch03.ex05 import identity_function
from lab_dl.dataset.mnist import load_mnist


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어옴
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 확인
    return network


def predict(network, test_set):
    """
    - 신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴

    예측을 하려면 test set(array들)을 주어야 함

    :param network: 신경망에서 사용되는 가중치/bias 행렬들을 저장한 dict
    :param test_set: 입력값을 가지고 있는 test set
    """
    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 은닉층에서 활성화 함수: sigmoid 함수
    a1 = test_set.dot(W1) + b1
    z1 = sigmoid(a1)  # 첫번째 은닉층 전파
    z2 = sigmoid(z1.dot(W2) + b2)  # 두번째 은닉층 전파
    # 출력층: z2 @ W3 + b3 값을 그대로 출력
    y = z2.dot(W3) + b3
    return identity_function(y)  # 출력층의 활성화 함수를 적용 후 리턴 : 예측값을 숫자로 찾고 싶을 때


def accuracy(test_label, y_test):
    """
    - 테스트 데이터 레이블과 테스트 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴

    레이블, 예측값 - 숫자 리턴

    :param test_label: 테스트 레이블
    :param y_test: 테스트 데이터 예측값
    """
    answer = 0
    for i in range(len(test_label)):
        if test_label[i] == y_test[i]:
            answer += 1
    return answer / len(test_label)


if __name__ == '__main__':
    # 신경망 생성
    network = init_network()
    
    # step 1. 
    # (학습 이미지 60,000 장 데이터, 학습 데이터 레이블), (테스트 이미지 데이터 세트, 테스트 데이터 레이블) 두개의 튜플 만들기
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    print('X_train shape : ', X_train.shape)
    print('y_train shape : ', y_train.shape)
    
    # step 2.
    y_pred = predict(network, X_test)

    # step 3.
    acc = accuracy(y_test, y_pred)


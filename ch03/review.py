"""
신경망을 행렬로 나타내기
"""
import pickle
import numpy as np
from ch03.ex01 import sigmoid
from ch03.ex11 import softmax
from dataset.mnist import load_mnist


def init_network():
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())  # keys 확인
    return network

def forword(network, X_test, batch_size):  # 이렇게까지는 필요 없다고 생각~
    # 가중치
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # 편향
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    result = []
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i: (i + batch_size)]
        result_list = []
        for x_i in X_batch:
            # 첫번째 은닉층 전파
            z1 = sigmoid(x_i.dot(W1) + b1)
            # 두번째 은닉층 전파
            z2 = sigmoid(z1.dot(W2) + b2)
            result_list.append(z2)
        result.append(result_list)

    # 출력층 전파 : X 자리에 n차원 행렬을 넣어주었다.
    y = softmax(z2.dot(W3) + b3)

    return y

def batch_size_predict(network, X_test, batch_size):
    y_hat = forword(network, X_test, batch_size)
    # y_pred = np.argmax(y_hat, axis=1)
    # return np.array(y_pred)




if __name__ == '__main__':
    # data bring in !
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 초기 객체 생성
    network = init_network()

    # X_test : 2차원 배열 parameter - predict 함수
    # mini_batch = 100 설정
    mini_batch = 100
    y = forword(network, X_test, 100)
    print(y[:5])
    y_pred = batch_size_predict(network, X_test, 100)
    print(y_pred)
"""
mini-batch
"""
import pickle

import numpy as np
from ch03.ex01 import sigmoid
from ch03.ex08 import init_network, accuracy
from dataset.mnist import load_mnist


def softmax(X):
    """
    X 에 들어올 수 있는 차원 (가정)
    1) X - 1차원 : [x_1, x_2, .. ]
    2) X - 2차원 : [[x_11, x_12, .. ],
                   [x_21, x_22, ...],
                   ...             ]]
    """
    dimension = X.ndim  # np.ndim : array 의 dimension 을 준다.
    if dimension == 1:
        m = np.max(X)  # 1차원의 최댓값
        x = X - m  # 최댓값을 빼면 전부 0보다 같거나 작다 <- exp 함수의 overflow 를 방지하기 위함!!
        y = np.exp(X) / np.sum(np.exp(X))
        # y = 배열 / 스칼라 -> broadcast 된다. (for 문 역할 but 성능 빠름)
    elif dimension == 2:
        # reshape
        # m = np.max(X, axis=1).reshape((len(X), 1))  # len(x) 행 - 2차원 배열은 1차원 배열들을 원소로 갖는 배열 ( 2차원 배열의 row 의 갯수)
        # X = X - m
        # sum = np.sum(np.exp(X), axis=1).reshape((len(X), 1))
        # y = np.exp(X) / sum

        # transpose 2
        Xt = X.T
        m = np.max(Xt, axis=0)
        Xt = Xt - m
        y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
        y = y.T

    return y


def forword2(network, x):
    """
    :param x: 2차원이라고 가정
    """
    # 가중치 행렬:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # bias 행렬:
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 첫번째 은닉층 전파(propagation) - 2차원인데 괜찮은지,,?
    z1 = sigmoid(x.dot(W1) + b1)
    # 두번째 은닉층 전파(propagation)
    z2 = sigmoid(z1.dot(W2) + b2)
    # 출력층 전파 - ex11에서 만든 softmax 를 넣는다 (softmax 지금 꺼 아니라도 가능)  -> ???
    y = softmax(z2.dot(W3) + b3)

    return y


def predict2(network, X_test):
    prediction = []
    for x_i in X_test:
        # 각 이미지를 신경망에 전파(통과)시켜 확률을 계산하는 과정 반복
        y_hat = forword2(network, x_i)
        # 가장 큰 확률의 index(예측값) 를 준다
        y_pred = np.argmax(y_hat)
        prediction.append(y_pred)
    return np.array(prediction)  # mini_batch에서는 사용하지 않는다


def mini_batch(network, X_test, batch_size):  # my answer before I arrived home + 오쌤 정답
    """
    :param network: set 되어 있는 신경망
    :param X_test: 테스트 데이터
    :param batch_size: 입력층에 들어갈 size
    """
    result = np.array([])
    for i in range(0, len(X_test), batch_size):
        # batch
        x_batch = X_test[i:(i+batch_size)]
        y_batch = forword2(network, x_batch)  # x_batch 를 X_test 로 주는 데 forword()를 써야대~ 그리고 축을 1로 주어야 해요!
        p = np.argmax(y_batch, axis=1)  # row 축으로 argmax 뽑는다.
        result = np.append(result, p)  #
    return result


def mini_batch(network, X_test, batch_size):  # other student's answer

    r = np.zeros(0)
    print(r)

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        X_batch_pred = predict2(network, X_batch)
        r = np.r_[r, X_batch_pred]  # row binding
        print(r)

    return r


def mini_batch(network, X, batch_size):  # 오쌤 정답 - X - X_test 인 거에는 변함이 없다
    y_pred = np.array([])
    # batch_size -> forward propagation
    for i in range(0, len(X), batch_size):
        x_batch = X_test[i:(i+batch_size)]
        y_hat = forword2(network, x_batch)  # shape : (batch size, 10) 인 2차원 배열
        # 각 row 에서의 최댓값의 인덱스 -> (batch_size, ) 의 1차원 배열
        predictions = np.argmax(y_hat, axis=1)  # np.array 로 출력되므로 y_pred 를 np.array([])로 만들었다
        y_pred = np.append(y_pred, predictions)
    return y_pred  # shape : (len(X), )


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == '__main__':
    # 1 차원 softmax 테스트
    np.random.seed(2020)
    a = np.random.randint(10, size=5)
    print('a : \n', a)
    # softmax 함수 특징 알기! - 함수 적용 (합하면 1, 크기 순서 그대로 유지)
    print(softmax(a))

    # 2 차원 softmax 테스트
    A = np.random.randint(10, size=(2, 3))  # 난수 0~ 10보다 작은 양의 정수들 중 (2, 3) 행렬
    print('A : \n', A)
    print(softmax(A))  # e-03 = 1/1000 = 0.001


    # 실습
    # dataset(train, test) loading
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    print('X_test.shape : \n', X_test.shape)  # 784 = 28*28
    print('y_test.shape : \n', y_test.shape)
    print('X_test 0th row : \n', X_test[0])
    print('y_test 0th row : \n', y_test[0])

    # 신경망 생성 (가중치 행렬) - init_network 안쓰고 바로 해버렸음
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print('network (keys) : \n', network.keys())
    print('W1 :\n', network['W1'].shape)
    print('W2 :\n', network['W2'].shape)
    print('W3 :\n', network['W3'].shape)

    batch_size = 100
    y_pred = mini_batch(network, X_test, batch_size)  # 정확도 계산을 위한 정답 모양 설정
    print('true 10 :\n', y_test[:10])
    print('pred 10 :\n', y_pred[:10])
    # 신경망에 보내는 이미지 한번에 100개 * for 문 100번 = 10000개 y_pred

    # 정확도 계산하여 출력
    print('Accuracy : \n', accuracy(y_test, y_pred))  # 속도 빨라짐
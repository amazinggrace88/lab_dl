"""
test 2층 신경망 테스트
"""
import random
import pandas as pd
import numpy as np
from lab_dl.ch05.ex10_twolayer import TwoLayerNetwork
from lab_dl.dataset.mnist import load_mnist
from sklearn.utils import shuffle

if __name__ == '__main__':
    np.random.seed(106)
    # MNINT 데이터 로드
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # 2 층 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
    epochs = 100  # 100번 반복
    batch_size = 100  # batch size is 100 - 한번에 학습시키는 입력 데이터 갯수
    learning_rate = 0.1
    iter_size = max(X_train.shape[0] // batch_size, 1)  # 6만개를 100개씩 input 해준다. - 전체 1세트 훈련 끝
    print(iter_size)  # batch_size = 60001 일 때에는 iter_size 가 정수가 안된다.. sol : max(, 1) 로 적어도 한 번 반복
    for i in range(iter_size):  # iter_size 만큼 반복
        # forward -> backward -> gradient 계산
        # gradient 안에 forward, backward 후 gradient 계산하는 과정 모두 있으므로
        # 처음 batch_size 갯수만큼의 데이터를 입력으로 하여 gradient 계산
        grad = neural_net.gradient(X_train[i*batch_size:i*batch_size+batch_size], y_train[i*batch_size:i*batch_size+batch_size])

        # 수정된 gradient 를 사용하여 가중치/편향 행렬들을 수정
        neural_net.params['W1'] -= learning_rate * grad['W1']
        neural_net.params['b1'] -= learning_rate * grad['b1']
        neural_net.params['W2'] -= learning_rate * grad['W2']
        neural_net.params['b2'] -= learning_rate * grad['b2']

    # 학습 완료된 loss 리턴
    loss = neural_net.loss(X_train, y_train)
    print('learned loss = ', loss)
    # 학습 완료된 acc 리턴
    acc = neural_net.accuarcy(X_train, y_train)
    print('learned accuarcy = ', acc)

    # shuffle_me
    # temp = [[x, y] for x, y in zip(X_train, y_train)]
    # random.shuffle(temp)
    # print(temp)
    # X_train = [n[0] for n in temp]
    # X_train = np.array(X_train)
    # y_train = [n[1] for n in temp]
    # y_train = np.array(y_train)

    # shuffle 2 번
    X_train, y_train = shuffle(X_train, y_train)



    for epoch in range(epochs):  # 반복할 때마다 학습 데이터 세트를 무작위로 섞는(shuffle) 코드를 추가
        for i in range(iter_size):
            grad = neural_net.gradient(X_train[(i*batch_size):(i*2*batch_size)], y_train[(i*batch_size):(i*2*batch_size)])

            # 수정된 gradient 를 사용하여 가중치/편향 행렬들을 수정
            for key in ('W1', 'W2', 'b1', 'b2'):
                neural_net.params[key] -= learning_rate * grad[key]

    # 학습 완료된 loss 리턴
    loss = neural_net.loss(X_train, y_train)
    print('learned loss = ', loss)
    # 학습 완료된 acc 리턴
    acc = neural_net.accuarcy(X_train, y_train)
    print('learned accuarcy = ', acc)

    # 각 epoch 마다 테스트 데이터로 테스트를 해서 accuracy를 계산
    # 100번의 epoch 가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.



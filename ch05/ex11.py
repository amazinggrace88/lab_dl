"""
test 2층 신경망 테스트
"""
import pickle
import random
import pandas as pd
import numpy as np
from lab_dl.ch05.ex10_twolayer import TwoLayerNetwork
from lab_dl.dataset.mnist import load_mnist
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(106)
    # MNINT 데이터 로드
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # 2 층 신경망 생성
    # hidden_size 는 hyperparameter (결정 가능)
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
    # batch size - 한번에 학습시키는 입력 데이터 갯수
    batch_size = 128
    # learning_rate 는 hyperparameter (결정 가능) - 학습률이 너무 커지면 발산 / 너무 작으면 시간 너무 오래 걸림
    learning_rate = 0.1
    # iter_size : 한번의 epoch(학습이 한 번 완료되는 주기)를 의미
    #           : 가중치/편향 행렬들이 한 번의 학습 주기(epoch)에서 변경되는 횟수 (6만개를 100개씩 input 해준다. - 전체 1세트 훈련 끝)
    iter_size = max(X_train.shape[0] // batch_size, 1)
    # batch_size = 60001 일 때에는 iter_size 가 정수가 안된다.. sol : max(, 1) 로 적어도 한 번 반복
    print(iter_size)

    # 1) 학습 1번 실행
    for i in range(iter_size):
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

    # 2) 학습 100번 실행
    # 학습 100번 반복
    epochs = 50
    train_losses = []  # 각 epoch 마다 학습 데이터의 손실을 저장할 리스트
    train_accuracies = []  # 각 epoch 마다 학습 데이터의 정확도를 저장할 리스트
    test_accuracies = []  # 각 epoch 마다 테스트 데이터의 정확도를 저장할 리스트

    for epoch in range(epochs):
        # 학습 데이터를 랜덤하게 shuffle
        # : 각 epoch 마다 shuffle 하여 학습시킴 ( like 문제의 순서를 바꿔서 시험 보는 것 )
        idx = np.arange(len(X_train))  # [0, 1, 2, ... 59999] 6 만개
        print(idx)
        # numpy 의 기능을 쓰기 위해 np.arange 를 만들었다.
        np.random.shuffle(idx)
        print(idx)  # index 를 섞어 X_train, y_train 의 인덱스로 넣는다.

        for i in range(iter_size):
            # batch_size 갯수만큼의 학습 데이터를 입력으로 하여 gradient 계산
            X_batch = X_train[idx[i*batch_size:(i+1)*batch_size]]
            Y_batch = y_train[idx[i*batch_size:(i+1)*batch_size]]
            gradients = neural_net.gradient(X_batch, Y_batch)

            # 수정된 gradient 를 사용하여 가중치/편향 행렬들을 수정
            for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradients[key]  # optimizor - SGD.update 메소드와 같은 형태

        # 각 epoch 마다 테스트 데이터로 테스트를 해서 accuracy 를 계산
        # 학습 완료된 loss 리턴
        train_loss = neural_net.loss(X_train, y_train)  # 전체 데이터를 모두 넣는다.
        train_losses.append(train_loss)
        # print('learned loss = ', train_loss)
        # 학습 완료된 acc 리턴
        train_acc = neural_net.accuarcy(X_train, y_train)
        train_accuracies.append(train_acc)
        # print('learned accuarcy = ', train_acc)
        test_acc = neural_net.accuarcy(X_test, y_test)
        test_accuracies.append(test_acc)
        # print('test accuarcy = ', test_acc)

    """
    ** hyperparameter 정리 **
    batch_size - 저 (계산 오래 걸림) -> 고 (정확도 떨어짐) : 2^승 으로 대입이 일반적
    learning_rate - 저 (시간 너무 오래 걸림) -> 고 (너무 커지면 발산)
    """
    # 100번의 epoch 가 끝났을 때, epochs-loss, epochs-accuracy 그래프를 그림.
    # epoch ~ loss 그래프
    x = range(epochs)  # x 의 범위
    plt.plot(x, train_losses)  # 한번의 학습마다 train_loss 를 추출.
    plt.title('loss - Cross Entropy')
    plt.show()
    # epoch ~ acc 그래프
    plt.plot(x, train_accuracies, label='train accuracy')
    plt.plot(x, test_accuracies, label='test accuracy')
    plt.show()
    
    # epoch ~ acc subplot 그래프
    fig = plt.figure()  # 텅 빈 초기 figure
    axis1 = fig.add_subplot(2, 1, 1)
    axis2 = fig.add_subplot(2, 1, 2)
    axis1.plot(x, train_accuracies)
    axis1.set_title('train acc - Cross Entropy')
    axis2.plot(x, test_accuracies)
    axis2.set_title('test acc - Cross Entropy')
    plt.show()

    # 신경망에서 학습이 모두 끝난 후 파라미터(가중치/편향 행렬)들을 파일에 저장
    # pickle 이용
    # 다음 학습 때 그 전 학습한 W, b 행렬을 가지고 다음 학습에 이용 가능
    with open('params.pickle', mode='wb') as f:
        pickle.dump(neural_net.params, f)
    # mode='b' : binary 형태로 저장 (객체 자체로 저장, 메모장으로 열리지 않음)

    # shuffle_another_method
    # temp = [[x, y] for x, y in zip(X_train, y_train)]
    # random.shuffle(temp)
    # print(temp)
    # X_train = [n[0] for n in temp]
    # X_train = np.array(X_train)
    # y_train = [n[1] for n in temp]
    # y_train = np.array(y_train)

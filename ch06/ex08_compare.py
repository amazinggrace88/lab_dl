"""
ex01~07 까지의 optimizer 알고리즘 성능 비교
파라미터 최적화 알고리즘 6개의 성능 비교
** 지표 **
1) 손실 함수 loss function
2) 정확도 함수 accuracy function

적은 학습횟수에 손실을 더 많이 적게 만드는 optimizer 알고리즘을 찾자.
"""
import matplotlib.pyplot as plt
import numpy as np
from lab_dl.ch05.ex10_twolayer import TwoLayerNetwork
from lab_dl.ch06.ex02_sgd import Sgd
from lab_dl.ch06.ex03_momentum import Momentum
from lab_dl.ch06.ex04_adagrad import AdaGrad
from lab_dl.ch06.ex05_adam import Adam
from lab_dl.ch06.ex06_rmsprop import RMSProp
from lab_dl.ch06.ex07_nesterov import Nesterov
from lab_dl.dataset.mnist import load_mnist

if __name__ == '__main__':
    # MNIST 손글씨 이미지 data bring in~
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # 최적화 알고리즘을 구현한 클래스의 인스턴스들을 dict 에 저장
    optimizers = dict()
    optimizers['SGD'] = Sgd()
    optimizers['Momentum'] = Momentum()
    optimizers['Adagrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSProp'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()

    # 은닉층 1개, 출력층 1개로 이루어진 신경망을 optimizers 갯수만큼 생성
    # 각 신경망에서 손실들을 저장할 dict 를 생성
    neural_nets = dict()
    train_losses = dict()
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
        train_losses[key] = []  # loss 들의 이력을 저장 - n 번째 학습횟수 때 loss 기록

    # 각각의 신경망을 학습시키면서 loss 를 계산하고 기록
    """
    how ? 
    6만개 중 랜덤으로 128개 뽑아서, gradient 계산 후 각 알고리즘 별 파라미터 업데이트.
    업데이트 된 파라미터로 손실 계산
    """
    iterations = 2_000  # 총 학습 횟수
    batch_size = 128  # 1번 학습에서 사용할 미니배치 크기
    train_size = X_train.shape[0]
    np.random.seed(108)
    for i in range(iterations):
        # 학습 데이터(X_train), 학습 레이블(Y_train) 에서 미니 배치 크기만큼 랜덤으로 데이터 선택
        # 128 개의 임의의 숫자들로 이루어진 리스트를 만드는 함수 : np.random.choice(임의의 숫자 범위, 리스트의 원소 갯수)
        batch_mask = np.random.choice(train_size, batch_size)
        # 학습에 사용할 미니배치 데이터 / 레이블 선택
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        # optimizer 별로 gradient 계산
        for key in optimizers:
            # 각각의 최적화 알고리즘을 사용해서 gradient 구하기
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # gradient 이용하여 파라미터 업데이트
            optimizers[key].update(neural_nets[key].params, gradients)
            # optimizer 별로 loss 계산
            loss = neural_nets[key].loss(X_batch, Y_batch)
            train_losses[key].append(loss)  # {'Sgd': [..], 'Momentum': [..] .. } 형태
        
        # 100번째 학습할 때마다 계산된 손실을 출력
        if i % 100 == 0:  # 101번째 학습이라면
            print(f'===== training # {i} =====')
            for key in optimizers:
                print(key, ':', train_losses[key][-1])  # key 값에 있는 리스트 원소 중 제일 마지막

    # graph 그리기 - 각각의 최적화 알고리즘별 손실 그래프
    x = np.arange(iterations)  # graph 의 x 좌표 - 학습 횟수로 두자.
    for key, losses in train_losses.items():  # train_losses 의 알고리즘별 손실
        plt.plot(x, losses, label=key)
    plt.title('Losses')
    plt.xlabel('# of training')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    """
    그래프 해석
    RMSProp - 학습량이 얼마 되지 않아도 급격하게 loss 가 떨어진다.
    
    """
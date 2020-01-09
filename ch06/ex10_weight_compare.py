"""
weight compare - 가중치 초기값에 따른 신경망 성능 비교 (MNIST 데이터 사용)
"""
from lab_dl.ch06.ex02_sgd import Sgd
from lab_dl.ch06.ex03_momentum import Momentum
from lab_dl.ch06.ex04_adagrad import AdaGrad
from lab_dl.ch06.ex05_adam import Adam
from lab_dl.ch06.ex06_rmsprop import RMSProp
from lab_dl.ch06.ex07_nesterov import Nesterov
from lab_dl.common.multi_layer_net import MultiLayerNet
import numpy as np
from lab_dl.dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# 실험 조건 세팅
weight_init_types = {
    'std=0.01': 0.01,
    'Xavier': 'sigmoid',  # 가중치 초깃값 N(0, sqrt(1/n))
    'He': 'relu'  # 가중치 초깃값 N(0, sqrt(2/n))
}

# 각 실험 조건 별로 테스트할 신경망을 생성 : 초기값 갯수만큼의 신경망을 생성한다.
neural_nets = dict()
train_losses = dict()
for key, type in weight_init_types.items():
    neural_nets[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, weight_init_std=type)
    # layer 4개(뉴런 갯수 100개씩)인 완전연결 다층 신경망.  weight_init_std=type 으로 값을 지정한다.
    train_losses[key] = []  # 빈 리스트 생성 - 실험(학습)하면서 손실값들을 저장

# MNIST 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
iterations = 2_000  # 학습 횟수
batch_size = 128  # mini batch
train_size = X_train.shape[0]  # =len(X_train)

# optimizer를 변경하면서 테스트
optimizer = {
    'Sgd': Sgd(),
    'Momentum': Momentum(),
    'Adagrad': AdaGrad(),
    'Adam': Adam(),
    'RMSProp': RMSProp()
    # 'Nesterov': Nesterov()
}

# 2,000번 반복
np.random.seed(109)
for name, value in optimizer.items():
    for i in range(iterations):
        # 미니 배치 샘플 랜덤 추출
        batch_mask = np.random.choice(train_size, batch_size)
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        # 테스트 신경망 종류마다 반복
        for key in weight_init_types:
            # gradient 계산
            gradients = neural_nets[key].gradient(X_batch, Y_batch)
            # 파라미터(W, b) 업데이트
            optimizer[name].update(neural_nets[key].params, gradients)
            # 손실(loss) 계산 -> 리스트 추가
            loss = neural_nets[key].loss(X_batch, Y_batch)
            train_losses[key].append(loss)
        # 손실 일부 출력
        if i % 1000 == 0:
            print(f'{name} ===training{i}===')
            for key, val in train_losses.items():
                print(key, ':', train_losses[key][-1])

# x축-반복 회수, y축-손실 그래프
x = np.arange(iterations)
for key, losses in train_losses.items():
    plt.plot(x, losses, labels=key)
plt.title('weight init compare')
plt.xlabel('# of training : iteration')
plt.ylabel('loss')
plt.legend()
plt.show()


# optimizer 1개 일 때 : Sgd
optimizer = Sgd()
for i in range(iterations):
    # 미니 배치 샘플 랜덤 추출
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]
    # 테스트 신경망 종류마다 반복
    for key, net in neural_nets.items():
        # gradient 계산
        gradients = net.gradient(X_batch, Y_batch)
        # 파라미터(W, b) 업데이트
        optimizer.update(net.params, gradients)
        # 손실(loss) 계산 -> 리스트 추가
        loss = net.loss(X_batch, Y_batch)
        train_losses[key].append(loss)
    if i % 100 == 0:
        print(f'===== iteration #{i} =====')
        for key, loss_list in train_losses.items():
            print(key, ':', loss_list[-1])


# x축-반복 회수, y축-손실 그래프
x = np.arange(iterations)
for key, losses in train_losses.items():
    plt.plot(x, losses, labels=key)
plt.title('weight init compare')
plt.xlabel('# of training : iteration')
plt.ylabel('loss')
plt.legend()
plt.show()

# optimizer 1개 일 때 : Adam
optimizer = Adam()
for i in range(iterations):
    # 미니 배치 샘플 랜덤 추출
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    Y_batch = Y_train[batch_mask]
    # 테스트 신경망 종류마다 반복
    for key, net in neural_nets.items():
        # gradient 계산
        gradients = net.gradient(X_batch, Y_batch)
        # 파라미터(W, b) 업데이트
        optimizer.update(net.params, gradients)
        # 손실(loss) 계산 -> 리스트 추가
        loss = net.loss(X_batch, Y_batch)
        train_losses[key].append(loss)
    if i % 100 == 0:
        print(f'===== iteration #{i} =====')
        for key, loss_list in train_losses.items():
            print(key, ':', loss_list[-1])  # 초기값이 중요한 역할을 할수도 있고, 아닐수도 있다.
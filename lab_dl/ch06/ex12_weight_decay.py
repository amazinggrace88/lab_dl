"""
과적합 (overfitting)
: train data 에 모델이 너무 학습을 시켜 학습되지 않은 데이터에 대해서 정확도가 떨어지는 현상.
** overfitting 이 나타나는 경우 **
    1) 학습 데이터가 적은 경우
    2) 파라미터가 너무 많아서 표현력(representational power)이 너무 높은 모델
        : y= ax1+bx2+r.. a,b,r와 같은 파라미터가 너무너무 많을 때

** overfitting 이 되지 않도록 학습 **
    1) regularization (규제, 정칙화)
    : L1 regularization/ L2 regularization - 책에서는 가중치 감소(weight decay) 라고 한다.
    손실(비용) 함수에 L1 규제(W) 또는 L2 규제(W**2)을 더해주어
    파라미터(W, b)를 갱신(업데이트)할 때, 파라미터가 더 큰 감소를 하도록 만드는 것.

    가중치 감소 (weight decay) 의 종류
    - L1 규제 - Loss = L + lambda(Λ) * ||W||
               ||X|| = sqrt((x1**2 + x2**2)) = 거리
               -> W = W - lr * (바뀐 Loss = dL/dW + lambda)
               -> 모든 가중치가 일정한 크기로 감소됨.

    - L2 규제 - Loss = L + lambda(Λ) * (1/2)||W||**2
              ||X||**2 = (x1**2 + x2**2) = 거리의 제곱
              (1/2) 는 미분에 의해 사라질 상수이므로 크게 신경쓰지 않아도 됨.
              -> W = W - lr * (바뀐 Loss = dL/dW + lambda * W)
              -> 가중치가 더 큰 값이 더 큰 감소를 일으킴.

    lambda 의 목표 : 일부러 lambda 를 통해 손실을 더 주기 위함. (페널티 역할)
    가중치 행렬의 제곱을 더해버리면, weight update 되는 정도가 감소된다. why?

    2) Dropout
    : 학습 중에 은닉층의 뉴런을 랜덤하게 골라서 삭제하고 학습시키는 방법.
    테스트를 할 때는 모든 뉴런을 사용함.

    overfitting 을 줄이는 전략은 학습 시의 정확도를 일부러 줄이는 것임.
    - 학습 데이터의 정확도와 테스트 데이터의 정확도 차이를 줄이기 위함.

"""
from lab_dl.ch06.ex02_sgd import Sgd
from lab_dl.common.multi_layer_net import MultiLayerNet
from lab_dl.dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(110)
# data bring in!
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# 신경망 생성
wd_rate = 0.15  # lambda
neural_net = MultiLayerNet(input_size=784,
                           hidden_size_list=[100, 100, 100, 100, 100],
                           output_size=10,
                           weight_decay_lambda=wd_rate)

# 학습 데이터 갯수를 300개로 제한 -> overfitting 만들기 위해서
X_train = X_train[:300]
Y_train = Y_train[:300]
X_test = X_test[:300]
Y_test = Y_test[:300]

# 에포크 : 모든 훈련(학습) 데이터가 1번씩 학습이 된 상태
epochs = 200
# minibatch : 1 번 forward 에 보낼 데이터 샘플 갯수 (신경망에 input 3번 = epoch 1번)
mini_batch = 100
train_size = len(X_train)
iter_per_epoch = int(max(train_size / mini_batch, 1))
# 학습하면서 테스트 데이터의 정확도를 각 epoch 마다 기록
train_accuracy = []
test_accuracy = []
# optimizer
optimizer = Sgd()

# training
for epoch in range(epochs):
    for i in range(iter_per_epoch):
        x_batch = X_train[(i * mini_batch):((i+1) * mini_batch)]
        y_batch = Y_train[(i * mini_batch):((i+1) * mini_batch)]

        gradients = neural_net.gradient(x_batch, y_batch)
        optimizer.update(neural_net.params, gradients)

    # train data 300개를 모두 계산하여 accuracy 를 준다.
    train_acc = neural_net.accuracy(X_train, Y_train)
    train_accuracy.append(train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    test_accuracy.append(test_acc)

    print(f'iteration #{epoch}: train={train_acc}, test={test_acc}')

x = np.arange(epochs)
plt.plot(x, train_accuracy, label='train')
plt.plot(x, test_accuracy, label='test')
plt.legend()
plt.show()

# lambda 값을 바꿔서 해보기
# Weight decay 신경망 생성
decay_neural_net = MultiLayerNet(input_size=784,
                           hidden_size_list=[100, 100, 100, 100, 100],
                           output_size=10,
                           weight_decay_lambda=0.5)

for i in range(epochs):
    mask = np.random.choice(len(X_train), mini_batch)
    pass
"""
배치 정규화(Batch Normalization)
: 신경망의 각 층에 미니 배치를 전달할 때마다 정규화를 실행하도록 강제하는 방법

**idea**
정규화: 가능한 모든 데이터가 일정하게 신경망에 전달되기 위해 사용함
신경망의 layer를 지날 때마다 N(0, 1)을 맞춰놓은 데이터가 정규화가 풀어지고, 어떤 데이터는 커지고 어떤 데이터는 작아질 것이다.
각 층마다 정규화를 하면 결과가 어떻게 나올까?

**impact** but this is case-by-case!
- 학습속도 개선
- 파라미터의 초깃값(W, b)에 크게 의존하지 않는다. (6.2 초깃값 이 필요가 없어진다)
- 과적합(overfitting)을 억제한다.

**result**
y = gamma * x + beta
hyperparameter - gamma & beta 추가된다.
gamma : 정규화된 mini-batch 를 scale-up/down (1보다 커지면 scale-up / 1보다 작으면 scale-down)
beta : 정규화된 mini-batch 를 이동시킴(bias 역할)
배치 정규화를 사용할 때에는 gamma & beta 초기값을 설정하고, 학습하면서 W, b 처럼 계속 갱신(update)해주어야 한다.

cf. nowadays 신경망에서 표준처럼 사용되고 있다.
"""
from lab_dl.ch06.ex02_sgd import Sgd
from lab_dl.common.multi_layer_net_extend import MultiLayerNetExtend
from lab_dl.common.optimizer import Momentum
from lab_dl.dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(110)

# p.213 그림 6-18 을 그리세요.
# Batch-Normalization 을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교하기

# MNIST data
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# Batch-Normalization 을 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size=784, 
                                    hidden_size_list=[100, 100, 100, 100, 100], 
                                    output_size=10, 
                                    weight_init_std=0.3,
                                    use_batchnorm=True)

# Batch-Normalization 을 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size=784, 
                                 hidden_size_list=[100, 100, 100, 100, 100], 
                                 output_size=10, 
                                 weight_init_std=0.3,
                                 use_batchnorm=False)

# mini-batch 20 번 학습시키면서 두 신경망에서 정확도(accuarcy)를 기록
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)
# 학습 시간을 줄이기 위해서 학습 데이터의 개수를 줄임.
X_train = X_train[:1000]  # 데이터 1000개만 사용 - batch size 쓰지 않았음..
Y_train = Y_train[:1000]

train_size = X_train.shape[0]
batch_size = 128
learning_rate = 0.01
iterations = 204

train_accuracies = []  # 배치 정규화를 사용하지 않는 신경망의 정확도를 기록
bn_train_accuracies = []  # 배치 정규화를 사용하는 신경망의 정확도를 기록

# optimizer = Sgd(learning_rate)  # 신경망 모양마다 optimizer 가 달라져야 한다.

# 파라미터 최적화 알고리즘이 SGD 가 아닌 경우에는 신경망 개수만큼 optimizer를 생성하자.
optimizer = Momentum(learning_rate)
bn_optimizer = Momentum(learning_rate)  # ?????

# 학습하면서 정확도의 변화를 기록
np.random.seed(110)
for i in range(iterations):
    # 미니 배치를 랜덤하게 선택(0~999 숫자들 중 128개를 랜덤하게 선택)
    mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[mask]
    y_batch = Y_train[mask]

    # 배치 정규화를 사용하지 않는 신경망에서 gradient를 계산.
    gradients = neural_net.gradient(x_batch, y_batch)
    # 파라미터 업데이트(갱신) - W(가중치), b(편향)을 업데이트
    optimizer.update(neural_net.params, gradients)  # neural_net.params 은 W, b만을 넘기고 있다. momentum 은 감마, 베타도 필요..
    # 업데이트된 파라미터들을 사용해서 배치 데이터의 정확도 계산
    acc = neural_net.accuracy(x_batch, y_batch)
    # 정확도를 기록
    train_accuracies.append(acc)

    # 배치 정규화를 사용하는 신경망에서 같은 작업을 수행.
    bn_gradients = bn_neural_net.gradient(x_batch, y_batch)  # gradient 계산
    bn_optimizer.update(bn_neural_net.params, bn_gradients)  # W, b 업데이트
    bn_acc = bn_neural_net.accuracy(x_batch, y_batch)  # 정확도 계산
    bn_train_accuracies.append(bn_acc)  # 정확도 기록

    print(f'iteration #{i}: without={acc}, with={bn_acc}')

# 정확도 비교 그래프
x = np.arange(iterations)
plt.plot(x, train_accuracies, label='Without BN')
plt.plot(x, bn_train_accuracies, label='Using BN')
plt.legend()
plt.show()






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
from lab_dl.dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# p.213 그림 6-18 을 그리세요.
# Batch-Normalization 을 사용하는 신경망과 사용하지 않는 신경망의 학습 속도 비교하기

# MNIST data
(X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

# Batch-Normalization 을 사용하는 신경망
bn_neural_net = MultiLayerNetExtend(input_size=784, 
                                    hidden_size_list=[100, 100, 100, 100, 100], 
                                    output_size=10, 
                                    weight_init_std=0.01, 
                                    use_batchnorm=True)

# Batch-Normalization 을 사용하지 않는 신경망
neural_net = MultiLayerNetExtend(input_size=784, 
                                 hidden_size_list=[100, 100, 100, 100, 100], 
                                 output_size=10, 
                                 weight_init_std=0.01, 
                                 use_batchnorm=False)

# mini-batch 20 번 학습시키면서 두 신경망에서 정확도(accuarcy)를 기록

mini_batch = 20
train_losses = []

for i in range(mini_batch):
    gradients = bn_neural_net.gradient(X_train[mini_batch], Y_train[mini_batch])
    params = bn_neural_net.params
    optimizer.update(params, gradients)
    loss = bn_neural_net.loss()
    train_losses.append(loss)

# 그래프 그리기
plt.plot(X_batch, train_losses)
plt.show()






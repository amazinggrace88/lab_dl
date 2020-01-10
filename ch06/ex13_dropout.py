import numpy as np
import matplotlib.pyplot as plt
from lab_dl.ch06.ex02_sgd import Sgd
from lab_dl.common.multi_layer_net_extend import MultiLayerNetExtend
from lab_dl.dataset.mnist import load_mnist

np.random.seed(110)
x = np.random.rand(20)
print('x = ', x)  # 0.0 ~ 0.9999 균등분포에서 뽑은 난수

# masking
mask = x > 0.5  # broadcasting
print(mask)
print(x * mask)
"""
x element 와 false(0) 를 곱하면 0
x element 와 true(1) 를 곱하면 x element
np.random.rand - uniform distribution 균일분포이므로
숫자가 커지면 커질수록 확률은 50% 에 가까워진다.
"""

"""mask 를 기반으로 dropout 해석하기"""
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):  # 50% 를 drop out 하겠다는 의미
        self.dropout_ratio = dropout_ratio
        self.mask = None  # 배열 중 몇 개를 뽑아낼 때 mask - (T, F ..) 를 사용한다.

    def forward(self, x, train_flg=True):  # train_flg : 학습중인지 알려주는 파라미터
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  # ratio 보다 큰 숫자만 mask
            return x * self.mask  # 숫자 * (true - 1, false - 0)
        else:
            return x * (1.0 - self.dropout_ratio)  # 원래 출력값 * (1-dropout ratio)

    def backward(self, dout):
        return dout * self.mask


if __name__ == '__main__':
    np.random.seed(110)
    # data
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # neural_net setting
    dropout_ratio = 0.1  # drop out 비율을 0.1 로 지정
    neural_net = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     use_dropout=True,
                                     dropout_ration=dropout_ratio)
    # parameter setting
    train_size = X_train.shape[0]
    mini_batch_size = 100
    epochs = 200
    iter_per_epoch = int(max(train_size / mini_batch_size, 1))
    optimizer = Sgd()

    train_accuracy = []
    test_accuracy = []

    # training ------------------> 다시 고쳐야됨!
    for epoch in range(epochs):
        for i in range(iter_per_epoch):
            x_batch = X_train[(i * mini_batch_size):((i+1) * mini_batch_size)]
            y_batch = Y_train[(i * mini_batch_size):((i+1) * mini_batch_size)]

            gradients = neural_net.gradient(x_batch, y_batch)
            optimizer.update(neural_net.params, gradients)  # 따로 변수 지정하지 않아도 괜찮다.

        train_acc = neural_net.gradient(X_train, Y_train)
        train_accuracy.append(train_acc)
        test_acc = neural_net.gradient(X_test, Y_test)
        test_accuracy.append(test_acc)

    # x - epochs /
    x = np.arange(epochs)
    plt.plot(x, train_accuracy, label='train')
    plt.plot(x, test_accuracy, label='test')
    plt.legend()
    plt.show()
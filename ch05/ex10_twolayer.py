"""
class - two layer 를 갖는 신경망 만들기
"""
from collections import OrderedDict
import numpy as np
from lab_dl.ch04.ex03 import cross_entropy
from lab_dl.ch05.ex05_relu import Relu
from lab_dl.ch05.ex07_affine import Affine
from lab_dl.ch05.ex08_softmax_loss import SoftmaxWithLoss
from lab_dl.dataset.mnist import load_mnist


class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """신경망 초기화 작업 - 모양 결정"""
        # weight_init_std=0.01 : 정규분포에서 난수들의 표준편차 - input 의 분포가 0에 가깝도록
        # hidden_size 와 output_size 의 관계 설정 검색해보자
        np.random.seed(106)

        # 가중치/편향 행렬 초기화 - list, dict 에 저장하여 반복문 쉽도록 함
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # layer 생성 / 초기화
        self.layers = OrderedDict()
        # ordereddict - dict 에 데이터가 추가된 순서가 유지되는 딕셔너리
        # why? init 에 저장된 dict 는 반복문에서 순서가 없이 랜덤으로 출력되기 때문에
        # layer 에 만들어둔 순서를 맞추어 출력하기 위해 ordereddict 를 사용한다.
        self.layers['affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = Relu()  # 파라미터를 넣을 것이 없다
        self.layers['affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()  # dict 에 넣지 않았다

    def predict_me(self, X):
        # softmax 직전의 값들은 예측값과 비슷할 것이다.(확률의 순위는 바뀌지 않는다)
        # 확률이 아닌, 값들의 순위를 가지고 예측
        X = self.layers['affine1'].forward(X)
        X = self.layers['relu'].forward(X)
        X = self.layers['affine2'].forward(X)
        return X

    def predict(self, X):
        # softmax 직전의 값들은 예측값과 비슷할 것이다.(확률의 순위는 바뀌지 않는다)
        # 확률이 아닌, 값들의 순위를 가지고 예측
        for layer in self.layers.values():
            X = layer.forward(X)  # 다음 에 넘겨줄 parameter 로 forward 값을 넘긴다.
        return X

    def loss(self, X, y_true):
        y_pred = self.predict(X)
        loss = self.last_layer.forward(y_pred, y_true)  # 바로 loss 를 계산하는 게 아니라, softmax 적용해야 함
        return loss

    def accuarcy(self, X, y_true):
        """
        X, y_true 가 2차원 배열(행렬)일 때
        정확도 = 예측이 실제값과 일치하는 갯수 / 전체 입력 데이터 갯수
        정확도는 손실과 반비례를 가진다.
        """
        y_pred = self.predict(X)
        predictions = np.argmax(y_pred, axis=1)  # 행렬일 때 예측값 계산
        trues = np.argmax(y_true, axis=1)  # one hot encoding - 1 의 위치를 가진 인덱스를 찾는다.
        acc = np.mean(predictions == trues)  # 일치 갯수 / 전체 갯수
        return acc

    def gradient(self, X, y_true):
        """
        입력데이터 X와 실제값(레이블) y_true 가 주어졌을 때,
        모든 레이어에 대해서 forward propagation 을 수행한 후
        backward propagation 하여 (오차역전파 방법을 이용) W1, W2, b1, b2 의 gradient 를 찾는다.
        """
        gradients = dict()
        # 가중치/편향 행렬에 대한 gradient 들을 저장할 딕셔너리

        self.loss(X, y_true)  # forward propagation 끝!

        # backward propagation
        dout = 1
        dout = self.last_layer.backward(dout)
        # 순서가 있는 dict 에서 value 들만 꺼내서 [affine1, relu, affine2] 를 리스트로 만든다. - 순서 바꾸기 위해서
        layers = list(self.layers.values())
        layers.reverse()  # 순서를 역순으로 바꿈 [affine2, relu, affine1]
        for layer in layers:
            dout = layer.backward(dout)
        
        # 모든 레이어에 대해서 역전파가 끝나면, 가중치/편향 행렬의 gradient 기울기 저장 가능
        # 합쳐지는 부분 (affine)에만 W, b 있음
        gradients['W1'] = self.layers['affine1'].dW
        gradients['b1'] = self.layers['affine1'].db
        gradients['W2'] = self.layers['affine2'].dW
        gradients['b2'] = self.layers['affine2'].db

        return gradients


if __name__ == '__main__':
    """ 모델이 제대로 만들어 졌는지 test """
    # bring the data
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # data shape
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # (60000, 784) -> 784 input size : 열이 784개
    # (60000, 10) -> 10 output size : 열이 10개

    # neural network
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)
    # hidden_size 를 늘리거나 줄여서 feature engineering 할 수 있다
    for key in neural_net.params:  # 가중치와 편향 저장한 params - 변수 이름처럼 사용하여 for 문 만듬
        print(key, '.shape :', neural_net.params[key].shape)
    for key in neural_net.layers:  # 각 계층을 저장한 layers
        print(key, ' :', neural_net.layers[key])  # 계층이어서 key 값만 나오게 한다. (shape no)
    print(neural_net.last_layer)  # softmaxwithloss

    # image 1 개
    y_pred = neural_net.predict(X_train[0])
    print('y_pred = ', np.argmax(y_pred))
    loss = neural_net.loss(X_train[0], y_train[0])
    print('loss = ', loss)

    # image 3 개
    y_pred = neural_net.predict(X_train[:3])
    print('y_pred = ', y_pred)
    print('y_pred = ', np.argmax(y_pred, axis=1))  # 행마다 가장 큰 값을 찾아야 함
    loss = neural_net.loss(X_train[:3], y_train[:3])
    print('loss = ', loss)  # loss 는 cross entropy 의 평균이므로 image 1개 일 때의 loss 와 비슷하다.
    accuarcy = neural_net.accuarcy(X_train[:3], y_train[:3])
    print('acc = ', accuarcy)
    # gradient test
    gradients = neural_net.gradient(X_train[:3], y_train[:3])
    # print('W1 \'s gradient shape = ', gradients['W1'].shape)
    # print('b1 \'s gradient shape = ', gradients['b1'].shape)
    # print('W2 \'s gradient shape = ', gradients['W2'].shape)
    # print('b2 \'s gradient shape = ', gradients['b2'].shape)
    for key in gradients:
        print(gradients[key].shape, end=' ')

"""
Simple Convolutional Neural Network (CNN)
"""
from collections import OrderedDict
from lab_dl.lab_dl.common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
import numpy as np
from lab_dl.lab_dl.dataset.mnist import load_mnist


class SimpleConvNet:
    """
    1st hidden layer : Convolution -> Relu -> Pooling
    2st hidden layer : Affine -> Relu
    output layer : Affine -> Softmax with loss

        input_dim: 입력 데이터 차원. MNIST인 경우 (1, 28, 28) -> 3 차원
        conv_param: Convolution 레이어의 파라미터(filter, bias)를 생성하기 위해
        필요한 값들
            필터 개수(filter_num),
            필터 크기(filter_size = filter_height = filter_width),
            패딩 개수(pad),
            보폭(stride)
        hidden_size: Affine 계층에서 사용할 뉴런의 개수
        output_size: 출력값의 원소의 개수. MNIST인 경우 10
        weight_init_std: 가중치(weight) 행렬을 난수로 초기화할 때 사용할 표준편차

    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        인스턴스 초기화 - 계층 생성, 변수들 초기화
        """
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))  # 사이즈 줄여준다. for Affine2 계층

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

        # 각 계층들 별 params
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)  # filter 의 수만큼
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        """모든 layer를 지나면서 forward"""
        for layer in self.layer.values():
            x = layer.forward(x)
            return x
    
    def loss(self, x, t):
        """forward 끝에서 softmaxwithloss 지나면서 손실 계산"""
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """accuracy"""
        y = self.predict(x, t)
        y = np.argmax(axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc


    def gradient(self, x, t):
        """backward 하면서 gradient 를 계산 -> w, b 업데이트"""
        # 순전파
        self.loss(x, t)
        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

if __name__ == '__main__':
    # 5000 장 데이터 학습 -> 테스트
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=False)
    print('X_train shape : ', X_train.shape)
    mini_batch_x_train = X_train[:5000]
    mini_batch_y_train = y_train[:5000]

    # SimpleConvNet 생성
    simpleconvnet = SimpleConvNet()
    # 학습 -> 테스트
    y_pred = simpleconvnet.predict(mini_batch_x_train)
    # print('', y_pred)
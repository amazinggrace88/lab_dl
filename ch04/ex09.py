"""
2층 신경망
"""
import numpy as np

from lab_dl.ch03.ex11 import forward2
from lab_dl.dataset.mnist import load_mnist


class TwoLayerNetwork():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):  # __init__ 에 변수 설정해야 함
        """
        입력이 input_size - ex) 784(28*28)개이고
        첫번째 층(layer)의 뉴런 갯수 hidden_size - ex) 32개,
        출력층의 뉴런갯수 output_size - ex) 10개
        - 가중치 행렬 : w1, w2
        - bias 행렬 : b1, b2
        을 난수로 생성
        """
        np.random.seed(1231)
        self.params = dict()  # weight / bias 행렬을 저장하는 딕셔너리

        # 입력층 만들기(1, 784) @ w1 (784, 32) = (1, 32) + b1(1, 32)
        # weight_init_std 의미 : 시작하는 값을 가능하면 0 근처의 값부터 시작하겠다 (모든 확률이 비슷하게 나오게 하기 위함)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        # z1 = (1, 32) @ W2 (32, 10)  = (1, 10) + b2(1, 10)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):  # predict2 처럼 주면 왜 안될까? - 왜 1개짜리 row 는 나오지 않을까..?
        """
        data - hidden layer - output layer - 예측값
        sigmoid(data.dot(W1) + b1) = z
        sigmoid(z.dot(W2) + b2) = y
        y return
        밑에서 만든 sigmoid, softmax 를 여기서 쓰자 --> why? 1차원 배열로 해야 할 때,
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        z1 = self.sigmoid(x.dot(W1) + b1)
        z2 = self.sigmoid(z1.dot(W2) + b2)
        y = self.softmax(z2)
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, X):  # 2 차원 np.ndarray 가능
        if X.ndim == 1:
            m = np.max(X)
            x = X - m
            y = np.exp(X) / np.sum(np.exp(X))
        elif X.ndim == 2:
            Xt = X.T  # 전치행렬 만듬
            m = np.max(Xt, axis=0)  # max 함수 구함
            Xt = Xt - m
            y = np.exp(Xt) / np.sum(np.exp(Xt), axis=0)
            y = y.T
        return y

    def accuarcy(self, x, y_true):
        """
        :param x: 예측값을 구하고 싶은 데이터 (2차원 배열)
        :param y_true: 실제 레이블 (2차원 배열)
        :return: 정확도
        """
        y_pred = self.predict(x)
        # 배열에서 최댓값을 갖는 인덱스를 리턴, axis 가 없을 때에는 배열 전체, axis 가 있을 때에는 배열의 1행 중에서 찾는다.
        predictions = np.argmax(y_pred, axis=1)
        true_values = np.argmax(y_true, axis=1)
        print('predictions = ', predictions)
        print('true_values = ', true_values)
        acc = np.mean(predictions == true_values)  # sum/len 으로 나누어도 괜찮다~
        return acc

        # cnt = 0
        # for i in range(len(y_true)):
        #     p = np.argmax(y_pred[i])  # axis = 1로 줄일 수 있다.
        #     if p == np.argmax(y_true[i]):
        #         cnt += 1
        # return cnt / len(y_true) --> 이거는 error 왜그러징?

    def loss(self, x, y_true):
        """정확도의 반대"""
        y_pred = self.predict(x)
        entropy = self.cross_entropy(y_true, y_pred)
        return entropy

    def cross_entropy(self, y_true, y_pred):
        """cross_entropy 만들어보기"""
        if y_pred.ndim == 1:
            # 1차원 배열인 경우, 행의 갯수가 1개인 2차원 배열로 변환 (1, y_pred.size(원소갯수))
            y_pred = y_pred.reshape((1, y_pred.size))
            y_true = y_true.reshape((1, y_true.size))
        # y_true 는 one hot encoding 되어 있다고 가정
        # y_true 에서 1 이 있는 컬럼 위치(인덱스) 를 찾음
        true_values = np.argmax(y_true, axis=1)  # argmax axis=1을 주어 1 차원/2 차원 모두 쓸 수 있도록 함
        n = y_pred.shape[0]  # row 의 갯수 : (row, column)
        rows = np.arange(n)  # [0, 1, 2, 3..] 과 같은 배열로 row indexing 을 했다.
        # y_pred[[0, 1, 2], [2, 2, 3]] ???
        log_p = np.log(y_pred[rows, true_values])  # 배열 row, 배열 true_values
        entropy = -np.sum(log_p) / n
        return entropy

        # 설명?
        # y_pred[[0, 1, 2], [3, 3, 9]]
        # => [y_pred[0, 3], y_pred[1, 3], y_pred[2, 3]]

    def gradients(self, x, y_true):
        # grad 행렬을 찾아 W 행렬을 변화시켜야 함
        # loss 함수가 gradients 안에 들어가기 때문에, 똑같이 x, y_true 가 필요함
        loss_fn = lambda W: self.loss(x, y_true)  # 최소화시킬 확률
        gradients = dict()  # W1, b1, W2, b2 의 gradient 를 저장할 딕셔너리
        for key in self.params:
            # W1 행렬을 gradient 안에 주면,
            gradients[key] = self.numerical_gradient(loss_fn, self.params[key])
        return gradients

    def numerical_gradient(self, fn, x):
        h = 1e-4  # 0.0001
        gradient = np.zeros_like(x)
        with np.nditer(x, flags=['c_index'], op_flags=['readwrite']) as it:
            while not it.finished:
                # multi_index 를 통해서 2차원 배열에서도 모두 똑같이 동작하도록 함
                i = it.multi_index
                ith_value = it[0]  # 원본 데이터를 임시 변수에 저장
                it[0] = ith_value + h  # 원본 데이터 + h 만큼 저장
                fh1 = fn(x)  # f(x + h)
                it[0] = ith_value - h  # 원본 데이터 - h 만큼 저장
                fh2 = fn(x)  # f(x - h)
                gradient[i] = (fh1 - fh2) / (2 * h)
                it[0] = ith_value
                it.iternext()
        return gradient


if __name__ == '__main__':
    # 신경망 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)

    # W1, W2, b1, b2 의 shape 확인
    print('W1 :', neural_net.params["W1"].shape, 'b1 :', neural_net.params["b1"].shape)
    print('W2 :', neural_net.params["W2"].shape, 'b2 :', neural_net.params["b2"].shape)

    # data 불러오기(mnist)
    (X_train, y_train), (X_test, y_test) = load_mnist(one_hot_label=True)

    # 신경망 클래스의 predict() 메소드 테스트
    y_pred = neural_net.predict(X_train)

    # X_train[0]를 신경망에 전파(propagate)시켜서 예측값 확인 --> 2 차원 array 라서 안되는 거 같음 우짜징.?
    y_pred0 = neural_net.predict(X_train[0])
    print('y_true0 =', y_train[0])
    print('y_pred0 =', y_pred0)

    # X_train[:5]를 신경망에 전파시켜서 예측값 확인
    y_pred_five = neural_net.predict(X_train[:100])
    print('y_pred = \n', y_pred_five)
    print('sum = ', np.sum(y_pred_five, axis=1))  # softmax 함수의 특징 : 행끼리 더해서 1

    # accuracy, loss 메소드 테스트
    print('accuracy = ', neural_net.accuarcy(X_train[:100], y_train[:100]))
    print('loss = ', neural_net.loss(X_train[:100], y_train[:100]))

    # gradient 메소드 테스트
    gradients = neural_net.gradients(X_train[:100], y_train[:100])
    for key in gradients:
        print(key, np.sum(gradients[key]))

    # 찾은 gradient 를 이용해서 weight/bias 행렬들을 업데이트
    lr = 0.1
    for key in gradients:
        neural_net.params[key] -= lr * gradients[key]

    # mini-batch 방법 (이해해보기)
    epoch = 1000
    for i in range(epoch):
        for i in range(10):
            gradients = neural_net.gradients(X_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100])

            for key in gradients:
                neural_net.params[key] -= lr * gradients[key]

    # 10번만 해도 시간이 엄청 오래 걸린다..

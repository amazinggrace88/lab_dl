"""
p.179 오차역전파법 구현
Affine, ReLU, SoftmaxWithLoss 클래스 들을 이용한 신경망 구현
"""
import numpy as np

from lab_dl.ch05.ex05_relu import Relu
from lab_dl.ch05.ex07_affine import Affine
from lab_dl.ch05.ex08_softmax_loss import SoftmaxWithLoss

np.random.seed(106)

# 입력 데이터 : (1, 2) shape 의 ndarray - 데이터 1개씩만 넘기는 ndarray
X = np.random.rand(2).reshape(1, 2)
print('X = ', X)  # image 1장 - 2차원 배열이지만 원소가 1개밖에 없는 2차원 배열 for mini-batch
# 실제값 (정답)
Y_true = np.array([1, 0, 0])
print('Y true = ', Y_true)

# 첫번째 은닉층(hidden layer)
# W(가중치), b(편향) 행렬
# 첫번째 은닉층의 뉴런 갯수 = 3개
# W1 shape : (2, 3)
# b1 shape : (3, ) why? input x = (1, 2) @ w = (2, 3) = (1, 3)
W1 = np.random.randn(2,  3)
b1 = np.random.rand(3)
print('W1 = ', W1)
print('b1 = ', b1)
# Affine 객체 + ReLU(활성화 함수)
affine1 = Affine(W1, b1)
relu = Relu()

# 출력층 (output layer)
# W(가중치), b(편향) 행렬
# W2 shape : (3, 3) why? input 3, output 3 이므로
# b3 shape : (3, )  why? 분류 레이블 3개(output 3)라고 설정
W2 = np.random.randn(3, 3)
b2 = np.random.rand(3)
print('W2 = ', W2)
print('b2 = ', b2)
# Affine 객체 + SoftmaxWithLoss(활성화함수-and output)
affine2 = Affine(W2, b2)
last_layer = SoftmaxWithLoss()

# 각 레이어들을 연결 : forward propagation
# 은닉층 (hidden layer)
Y = affine1.forward(X)
print('Y shape = ', Y.shape)  # 3 개 원소 나와야 함
Y = relu.forward(Y)  # 변수를 공유해도 되요~
print('Y shape after Relu = ', Y.shape)
# 출력층 (output layer)
Y = affine2.forward(Y)
print('Y shape after affine2 = ', Y.shape)
loss = last_layer.forward(Y, Y_true)
print('Loss = ', loss)

"""
last layer 의 의미
- loss : loss = cross_entropy(self.y_pred, self.y_true) 계산 하여 숫자 1개로 리턴한다.
- class SoftmaxWithLoss 의 __init__ 에도 self.y_pred 저장 (출력 가능) - last_layer 에 저장
"""
# 예측값 (self.y_pred) 출력
print('y_pred = ', last_layer.y_pred)  # [[0.22573711 0.2607098  0.51355308]]

# gradient 계산 : backward propagation
# 출력층 (output layer)
learning_rate = 0.1
dout = last_layer.backward(1)  # default 1 임의설정
print('dout shape to Affine = ', dout.shape)  # forward 된 모양과 backward 된 모양은 똑같아야 함
dout = affine2.backward(dout)
print('dout shape to ReLU = ', dout.shape)  # W2, b2 의 gradient 도 저장되어 있음
print('dW2 = ', affine2.dW)  # W2 의 기울기
print('db2 = ', affine2.db)

# 은닉층 (hidden layer)
dout = relu.backward(dout)
print('dout shape to Affine1 = ', dout.shape)
dout = affine1.backward(dout)
print('dout shape to X = ', dout.shape)  # 쓸모없음
# 목적 : dW1, db1 를 찾는다 - W, b 의 기울기 행렬
print('dW1 = ', affine1.dW)
print('db1 = ', affine1.db)

# 가중치 행렬과 편향 행렬을 수정하기 - 매개변수 갱신
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db

# 수정된 가중치/편향 행렬들을 이용해서 다시 forward propagation
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('Loss 2 = ', loss)
print('y_pred 2 = ', last_layer.y_pred)  # [[0.29602246 0.25014373 0.45383381]] : [1, 0, 0]에 가깝게 변화

# 과제
print()
# mini_batch - X 를 행렬 (2차원 모양) 으로 만들자
X = np.random.rand(3, 2)  # 이미지 데이터를 3장씩 forward
print('X_assignment = ', X)
Y_true = np.identity(3)  # 대각행렬
print('Y_true = ', Y_true)
# forward 1
W1 = np.random.rand(2, 3)
b1 = np.random.rand(3)
Y = affine1.forward(X)
print('Y shape affine 1 = ', Y.shape)
Y = relu.forward(Y)
print('Y shape relu = ', Y.shape)
W2 = np.random.rand(3, 3)
b2 = np.random.rand(3)
Y = affine2.forward(Y)
print('Y shape affine 2 = ', Y.shape)
loss = last_layer.forward(Y, Y_true)
print('Loss = ', loss)
print('y_pred = ', last_layer.y_pred)

# backward
dout = last_layer.backward(1)
dout = affine2.backward(dout)
print('dout shape affine 2 = ', dout.shape)
print('W2 gradient matrix = ', affine2.dW)
print('b2 gradient matrix = ', affine2.db)
dout = relu.backward(dout)
print('dout shape relu = ', dout.shape)
dout = affine1.backward(dout)
print('dout shape affine 1 = ', dout.shape)  # (3, 2) why? 처음 input 이 (3, 2) 였으므로 ..
print('W1 gradient matrix = ', affine1.dW)
print('b1 gradient matrix = ', affine1.db)

# 매개변수 변환
learning_rate = 0.01
W1 -= learning_rate * affine1.dW
b1 -= learning_rate * affine1.db
W2 -= learning_rate * affine2.dW
b2 -= learning_rate * affine2.db

# forward 2 - loss 비교하기
Y = affine1.forward(X)
Y = relu.forward(Y)
Y = affine2.forward(Y)
loss = last_layer.forward(Y, Y_true)
print('loss 2 = ', loss)


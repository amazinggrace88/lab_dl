"""
Affine 계층
"""
import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W  # weight matrix
        self.b = b  # bias matrix
        self.X = None  # 입력 행렬을 저장할 field(변수)
        self.dW = None  # W 행렬 gradient -> W = W - lr * dW 에 사용됨
        self.db = None  # b 행렬 gradient -> b = b - lr * db 에 사용됨
        
    def forward(self, X):
        self.X = X  # backward 할 때 X 를 기억할 수 있도록 저장하자
        out = X.dot(self.W) + self.b
        return out

    def backward(self, dout):
        # dout - d-out 으로 X 방향으로 리턴 , W 방향으로 리턴, b 방향으로 리턴하는 부분이 있어야 한다.
        # 우리의 목적은 W, b 를 변화시키는 것 -> GD를 이용하여 W, b 를 fitting 시킬 때 사용 -> self.을 사용하여 저장한다.
        # b 행렬 방향으로의 gradient 있어야 함
        self.db = np.sum(dout, axis=0)  # 전파된 dout 에서 column 끼리 더한다.
        # z 행렬 방향으로의 gradient -> 1) W 방향 2) X 방향으로 나뉨
        self.dW = X.T.dot(dout)
        dX = dout.dot(W.T)

        return dX
        
        
if __name__ == '__main__':
    np.random.seed(103)

    X = np.random.randint(10, size=(2, 3))  # 입력 행렬 (ex_image data)
    W = np.random.randint(10, size=(3, 5))  # (2, 마음대로)
    b = np.random.randint(10, size=5)  # 마지막에 더해주는 bias 행렬

    affine = Affine(W, b)  # affine 클래스의 객체 생성

    Y = affine.forward(X)  # forward
    print('Y = ', Y)  # affine 의 출력값 -> 활성화 함수에 보낼 준비 끝!

    dout = np.random.randn(2, 5)  # 오차
    dX = affine.backward(dout)
    print('dX = ', dX)  # y이 1만큼 바뀔 때의 변화율을 알려준다.
    print('dW = ', affine.dW)  # affine 클래스가 가지고 있는 dW 변수 (. : 참조연산자)
    print('db = ', affine.db)  # affine 클래스가 가지고 있는 db 변수 (. : 참조연산자)


if __name__ == '__main__':
    pass
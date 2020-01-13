"""
sigmoid 함수: y = 1 / (1 + exp(-x))
dy/dx = y(1-y) 증명

sigmoid 뉴런을 작성(forward, backward)
layer 1 - (1 + exp(-x)) : AddLayer
layer 2 - 1 / x : MultiplyLayer
"""
import numpy as np
from lab_dl.ch05.ex01_basic_layer import MultiplyLayer, AddLayer

class UndermultiplyLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x  # 필드변수에 저장
        return 1 / x

    def backward(self, delta):
        dx = delta * (- 1 / (self.x ** 2))
        return dx

class ExponentialLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = np.exp(x)  # 필드변수에 저장
        return np.exp(x)

    def backward(self, delta):
        dx = delta * self.x  # 지수함수 미분해도 지수함수
        return dx


if __name__ == '__main__':
    # x 설정
    np.random.seed(12)
    x = np.random.rand()
    print('x = ', x)

    # 1. forward
    print('---forward---')
    first_layer = MultiplyLayer()
    first = first_layer.forward(x, -1)
    print(f'{first} = {x} * {-1}')

    second_layer = ExponentialLayer()
    second = second_layer.forward(first)
    print(f'{second} = np.exp(-x) = {np.exp(-x)}')

    third_layer = AddLayer()
    third = third_layer.forward(second, 1)
    print(f'{second} + 1 = {third}')

    fourth_layer = UndermultiplyLayer()
    fourth = fourth_layer.forward(third)
    print(f'1 / {third} = {fourth}')
    print()

    # 2. backward
    # answer : y(1-y)
    print('---backward---')
    print('y is = ', fourth)
    print('계산한 answer = ', fourth*(1-fourth))
    delta = 1.0
    dfourth = fourth_layer.backward(delta)
    print(f'dfourth = {dfourth}')
    dthree, add_one = third_layer.backward(dfourth)
    print('d_three = ', dthree)
    print('add_one = ', add_one)
    dtwo = second_layer.backward(dthree)
    print('d_two = ', dtwo)
    answer, add_minus_one = first_layer.backward(dtwo)
    print('증명한 answer = ', answer)

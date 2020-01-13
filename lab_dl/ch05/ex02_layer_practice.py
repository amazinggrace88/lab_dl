"""
f(x, y, z) = (x + y) * z
x = -2
y =  5
z = -3 에서의
df/dx
df/dy
df/dx 의 값을 구하라. (ex01 에서 구현한 MultiplyLayer 와 AddLayer 클래스를 이용)
numerical_gradient 함수를 가져와 계산 결과를 비교하기
"""
from lab_dl.ch04.ex05 import _numerical_gradient
from lab_dl.ch05.ex01_basic_layer import MultiplyLayer, AddLayer


multi_layer = MultiplyLayer()
add_layer = AddLayer()

# 1. forward
x, y, z = -2, 5, -4
q = add_layer.forward(x, y)
print('x_plus_y = ', q)
f = multi_layer.forward(q, z)
print('(x + y) * z = ', f)

# 2. backward
# - 주의점 : forward 에서 만든 layer 를 써야 한다.
#           즉, 생성자를 또다시 부르면 안된다.
delta = 1.0
dq, dz = multi_layer.backward(delta)
print('d/dq = ', dq)
print('d/dz = ', dz)
dx, dy = add_layer.backward(dq)
print('d/dx = ', dx)
print('d/dy = ', dy)


# _numerical_gradient 함수를 가져와 계산 결과를 비교하기
# 직접 계산
def f(x, y, z):
    return (x + y) * z

h = 1e-12
dx = (f(-2 + h, 5, -4) - f(-2 - h, 5, -4)) / (2 * h)
print('df/dx = ', dx)
dy = (f(-2, 5 + h, -4) - f(-2, 5 - h, -4)) / (2 * h)
print('df/dy = ', dy)
dz = (f(-2, 5, -4 + h) - f(-2, 5, -4 - h)) / (2 * h)
print('df/dz = ', dz)


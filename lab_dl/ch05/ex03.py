"""
교재 p.160 그림 5-15의 빈칸 채우기.
apple = 100원, n_a = 2개
orange = 150원, n_o = 3개
tax = 1.1
라고 할 때, 전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하세요.
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값들도 각각 계산하세요.
"""

from lab_dl.ch05.ex01_basic_layer import MultiplyLayer, AddLayer
# 1. 초기 설정
apple, n_a, orange, n_o, tax = 100, 2, 150, 3, 1.1

# 2. forward
apple_layer = MultiplyLayer()  # layer 1개 생성 = 뉴런 1개 생성
a = apple_layer.forward(apple, n_a)
print('apple * n_a = ', a)

orange_layer = MultiplyLayer()
o = orange_layer.forward(orange, n_o)
print('orange * n_o = ', o)

add_layer = AddLayer()
g = add_layer.forward(a, o)
print('a * n_a + o * n_o = ', g)

total_layer = MultiplyLayer()
f = total_layer.forward(g, tax)
print('(a * n_a + o * n_o) * tax = ', f)

# 3. backward
delta = 1.0
df, dtax = total_layer.backward(delta)
print('df : ', df)
print('dtax : ', dtax)  # d/dt = tax 가 1 높아질 때의 전체 가격의 변화량

da, do = add_layer.backward(df)
print('da : ', da)
print('do : ', do)

dapple, dn_a = apple_layer.backward(da)
print('dapple : ', dapple)  # df/dapple
print('dn_a : ', dn_a)  # df/dn_a

dorange, dn_o = orange_layer.backward(do)
print('dorange : ', dorange)  # df/dorange
print('dn_o = ', dn_o)  # df/dn_o
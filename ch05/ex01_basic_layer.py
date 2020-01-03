"""
basic layer -
f = x * y * t 일 때
"""

class MultiplyLayer:
    def __init__(self):
        # forward 메소드가 호출될 때 전달되는 입력값을 저장하기 위한 변수
        # backward 메소드가 호출될 때 사용되기 위함
        self.x = None
        self.y = None

    def forward(self, x, y):
        # 2개 input(parameter), 1개 output
        self.x = x  # 필드변수에 저장
        self.y = y  # 필드변수에 저장
        return x * y  # output

    def backward(self, delta_out):
        # 1개 input(parameter) - delta_out, 2개 output
        # backward 할 때 노드는 forward 후 자신의 연산을 기억하고 있어야 한다
        # backward 로 미분할 때 forward 의 연산을 알아야 하기 때문
        dx = delta_out * self.y  # x 쪽으로 진행되는 방향
        dy = delta_out * self.x  # y 쪽으로 진행되는 방향
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, delta_out):
        dx = delta_out * 1
        dy = delta_out * 1
        return dx, dy


if __name__ == '__main__':
    # MultiplyLayer 객체 생성
    apple_layer = MultiplyLayer()
    apple = 100  # 사과 한 개의 가격 : 100 원
    n = 2  # 사과 갯수 : 2 개
    apple_price = apple_layer.forward(apple, n)  # 순방향 전파
    print('사과 2개 가격 : ', apple_price)

    # tax_layer 를 MultiplyLayer 객체로 생성
    # tax = 1.1 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax_layer = MultiplyLayer()
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('총 가격 : ', total_price)

    """
    MultiplyLayer 객체를 새로 생성하는 이유:
    노드를 하나 더 생성해야 하기 때문에
    """
    # backward propagation(역전파)
    delta = 1.0  # 변화량
    dprice, dtax = tax_layer.backward(delta)
    print('dprice : ', dprice)
    print('dtax : ', dtax)  # df/dt
    dapple, dn = apple_layer.backward(dprice)

    # 사과 갯수가 1 증가하면 전체 가격은 얼마 증가할까? - df/dn
    print('dn = ', dn)
    # 사과 가격이 1 증가하면 전체 가격은 얼마 증가할까? - df/da
    print('da = ', dapple)
    # 사과 소비세가 1 증가하면 전체 가격은 얼마 증가할까? - df/dt : tax 변화율에 대한 전체 가격 변화율
    print('dtax = ', dtax)

    """
    MultiplyLayer + AddLayer test
    """
    add_layer = AddLayer()
    x = 100
    y = 200
    dout = 1.5
    f = add_layer.forward(x, y)
    print('f :', f)
    dx, dy = add_layer.backward(dout)
    print('dx = ', dx)
    print('dy = ', dy)

    orange_layer = AddLayer()
    orange_price = apple_layer.forward(150, 3)
    print('orange_price = ', orange_price)
    total_price = orange_layer.forward(orange_price, apple_price)
    print('오렌지와 사과의 총 가격 = ', total_price)
    total_tax_price = tax_layer.forward(total_price, tax)
    print(total_tax_price)
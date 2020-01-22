"""
instance 를 호출한다는 것
"""


class Foo:  # 클래스
    def __init__(self, init_val=1):
        print('__init__ 호출')
        self.init_val = init_val


class Boo:
    def __init__(self, init_val=1):
        print('__call__ 호출')
        self.init_val = init_val

    def __call__(self, n):  # 함수 이름 __call__ : 인스턴스를 호출 가능하게 하기 위함
        self.init_val *= n
        return self.init_val


if __name__ == '__main__':
    # Foo 클래스의 인스턴스를 생성 - 생성자를 호출
    foo = Foo()
    # 인스턴스 = 생성자() : 생성자는 heap 에 메모리 확보, 메모리에 데이터+함수들, 즉 클래스 저장
    print('init_val = ', foo.init_val)  # . 은 레퍼런스 연산자
    # foo(100) : 인스턴스는 호출이 가능하지 않은 객체(not callable)

    # Boo 클래스의 인스턴스 생성
    boo = Boo()
    print('boo init_val = ', boo.init_val)
    boo(5)  # 인스턴스를 함수처럼 호출하는 것이 가능하다. why? __call__이 있으면 instance 가 가지고 있는 call 을 호출한다.
    print('boo init_val = ', boo.init_val)

    """
    인스턴스 호출: 인스턴스 이름을 마치 함수 이름처럼 사용하는 것.
    클래스에 정의된 __call__ 메소드를 호출하게 됨.
    클래스에서 __call__을 작성하지 않은 경우에는 인스턴스 호출을 사용할 수 없음.
    """

    # callable : __call__ 메소드를 구현한 객체
    print('foo : ', callable(foo))
    print('boo : ', callable(boo))

    print()
    boo = Boo(2)
    x = boo(2)
    print('x = ', x)  # x = 4
    x = boo(x)
    print('x = ', x)  # x = 16

    print()
    input = Boo(1)(5)
    print(input)
    # why ? 5 : 변수를 함수 이름처럼 호출 =Boo(1)까지 생성자 호출 =인스턴스를 함수처럼 호출
    # boo = Boo(1), boo(5) 를 한줄로 쓴 것
    x = Boo(5)(input)
    print(x)
    x = Boo(5)(x)
    print(x)

    """
    클래스 안 메소드 호출
    """
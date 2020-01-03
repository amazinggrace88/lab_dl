"""
numpy.nditer 객체: 반복(for, while) 을 쉽게 쓰게 도와주는 객체
"""
import numpy as np

# code 를 줄여줄 수 있는 객체
np.random.seed(1231)
a = np.random.randint(100, size=(2, 3))
print(a)

# 40, 21, 5, 52, 84, 39
# for - 2번 사용
# for 문의 장점 : 간단하다
# for 문의 단점 : 인덱스와 값을 같이 쓰려면 enumerate(a) 를 주어야 한다.
for row in a:
    for x in row:  # 1 차원 배열에서 하나씩 꺼낸다
        print(x, end=' ')
print()

# while - 2번 사용
# while 문의 장점 : 인덱스로 접근 가능하므로 짝수번째 행만 찾는 행위 가능해짐 ( a[row, element] )
row = 0
while row < a.shape[0]:
    element = 0
    while element < a.shape[1]:
        print(a[row, element], end=' ')
        element += 1
    row += 1
print()

"""
iterator 사용방법

iterator 개념
- 하나의 row 가 끝나고 나면 다음 row 의 element로 넘어가는 역할
- 배열 a 에 있는 원소를 순서대로 꺼내줌
- 행 번호 증가시키기 전에 열번호를 증가시킨다.
- c_index : c 라는 언어는 (1, 1) -> (1, 2) -> (2, 1) -> (2, 2) 순으로 이동
- f_index : 포트런이라는 언어는 (1, 1) -> (2, 1) -> (1, 2) -> (2, 2) 순으로 이동  

"""
# with as 와 같이 쓰면 좋다 (자동 close)가 필요할 때 - for 1번 사용
with np.nditer(a) as iterator:  # nditer 클래스 객체 생성
    for val in iterator:
        print(val, end=' ')
print()

# iterator 를 while 문에서 사용하는 방법
with np.nditer(a, flags=['multi_index']) as iterator:
    # 공식 : 반복이 끝났으면 true, 반복이 끝나지 않으면 false + not
    #       = iterator 의 반복이 끝나지 않았으면
    while not iterator.finished:
        # multi_index 메소드를 쓰기 위해 flags 에 값을 준다.
        i = iterator.multi_index
        print(f'{i}, {a[i]}', end=' ')
        # 다음 순서로 바꾸라는 의미
        iterator.iternext()
print()

# c_index : 2 차원 배열에 차례대로 순서를 부여하는 것 (flatten과 비슷 ->)
with np.nditer(a, flags=['c_index']) as iterator:
    while not iterator.finished:
        i = iterator.index   # ??
        print(f'[{i}]{iterator[0]}', end=' ')  # 값을 변경하는 것은 불가능
        iterator.iternext()

a = np.arange(6).reshape((2, 3))
print(a)
with np.nditer(a, flags=['multi_index']) as it:
    while not it.finished:
        a[it.multi_index] *= 2
        it.iternext()
print(a)

a = np.arange(6).reshape((2, 3))
with np.nditer(a, flags=['c_index'], op_flags=['readwrite']) as it:
    while not it.finished:
        it[0] *= 2  # error : output array is read-only : sol) op_flags 필요함
        it.iternext()
print(a)

"""
iterator 의 장점
1차원, 2차원 등 차원이 다를때에도
한번에 코드를 만들 수 있다
"""

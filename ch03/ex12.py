"""
보충설명
1) for 문 범위
2) np.append()
"""
import numpy as np

# 1)
a = np.arange(10)
print('a :\n', a)
size = 3
for i in range(0, len(a), size):
    print(a[i:(i+size)])
# size 의 마지막은 9 ..
# a[9:9+3=12] 인 인덱스로 자르면 error 나지 않는다 why? 범위이므로
# a[12] 를 집으면 error 가 난다.

# 2) np.append()
# list 의 append
b = []
c = [1, 2, 3]
b.append(c)
print(b)  # c 가 b 의 하나의 원소로 추가 (2차원 - [[]] )

b = np.array([])
c = np.array([1, 2, 3])
b = np.append(b, c)
print(b)  # c 가 b 뒤에 가서 붙음, 즉 합쳐짐 (1차원 배열 - [] )

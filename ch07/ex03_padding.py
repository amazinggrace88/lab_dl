"""
padding
x pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)

** parameter 설명 **
1. x : 패딩을 넣을 배열
2. pad_width : pad 할 갯수
- pad_width= a 숫자라면, 양쪽 끝 0 넣음
- pad_width= (a, b) 튜플이라면, 앞쪽 a개, 뒤 b개 0 넣음 (before padding, after padding)
3. mode : padding 을 넣을 때 줄 숫자
- constant : 상수. constant_values=상수 를 넣을 것이다.
4. constant_values : 상수로 지정할 값
"""
import numpy as np

if __name__ == '__main__':
    np.random.seed(113)

    # 1 차원 ndarray
    x = np.arange(1, 6)
    print('x = ', x, sep='\n')
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)
    print('x padding / pad_width=1 =', x_pad, sep='\n')
    x_pad = np.pad(x, pad_width=(2, 3), mode='constant', constant_values=0)
    print('x padding / pad_width=(2, 3) =', x_pad, sep='\n')
    x_pad = np.pad(x, pad_width=2, mode='minimum')
    print('x padding / pad_width=2 =', x_pad, sep='\n')

    # 2 차원 ndarray
    x = np.arange(1, 10).reshape((3, 3))
    print('x = ', x, sep='\n')
    x_pad = np.pad(x, pad_width=1, mode='constant', constant_values=0)  # 축 상관없이 before padding, after padding 1씩
    print('x : 2d array padding / pad_width=1 =', x_pad, sep='\n')

    # axis=0 방향 before padding = 1
    # axis=0 방향 after padding = 2
    # axis=1 방향 before padding = 1
    # axis=1 방향 after padding = 2
    x_pad = np.pad(x, pad_width=(1, 2), mode='constant', constant_values=0)
    print('x : 2d array padding / pad_width=(1, 2) =', x_pad, sep='\n')

    # axis=0 방향 before padding = 1
    # axis=0 방향 after padding = 2
    # axis=1 방향 before padding = 3
    # axis=1 방향 after padding = 4
    x_pad = np.pad(x, pad_width=((1, 2), (3, 4)), mode='constant', constant_values=0)
    print('x : 2d array padding / pad_width=((1, 2), (3, 4)) =', x_pad, sep='\n')
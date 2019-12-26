"""
softmax 함수의 문제점과 해결책
"""
from lab_dl.ch03.ex05 import softmax
import numpy as np


if __name__ == '__main__':
    # 1 차원 배열
    x = np.array([1, 2, 3])
    s = softmax(x)
    print(s)

    # 2 차원 배열
    X = np.array([[1, 2, 3],  # image 1 장 - [] 리스트 안에서의 max 해야됨
                  [4, 5, 6]])
    s = softmax(X)
    print(s)  # 실행이 되긴 되는데, 값이 이상함.. 1~6 까지 더하고, 분자가 1이 됨 (max)

    x = [1, 2, 3]
    print('softmax =', softmax(x))

    x = [1e0, 1e1, 1e2, 1e3]  # [1, 10, 100, 1000]
    print('x =', x)
    print('softmax =', softmax(x))

    # numpy broadcast (브로드캐스트)
    # numpy array 의 축(axis) 에 대한 설명
    # axis = 0 : row 의 인덱스가 증가하는 축
    # axis = 1 : column 의 인덱스가 증가하는 축

    # array 와 scalar 간의 브로드캐스트 (scalar 를 array 에 맞춰서 복사)
    x = np.array([1, 2, 3])
    print('x = ', x)
    print('x + 10 = ', x + 10)

    # 2차원 array 와 1차원 array 간의 브로드캐스트
    X = np.arange(6).reshape((2, 3))
    print('X.shape : ', X.shape)
    print('X = \n', X)

    a = np.arange(1, 4)
    print('a.shape : ', a.shape)
    print('a = \n', a)
    # 브로드캐스트가 가능한 경우
    # (2, 3) = (3,) 3이 같고, array 의 열(,) 이 없을때
    print('X + a = \n', X + a)  # 1차원 array 의 행을 2 로 바꿔주었다

    b = np.array([10, 20]).reshape((2, 1))
    # 브로드캐스트가 가능하지 않은 경우
    # (2, 3) = (2, ) 3 != 2 이므로 한 방향으로 복사를 할 수 없어서 불가능 (필기 보기)
    print('b shape :', b.shape)
    # print(X + b) ValueError: operands could not be broadcast together with shapes (2,3) (2,)
    print('X + b = \n', X + b)

    # 브로드캐스트가 가능한 경우
    # (2, 3) = (2, 1) 2가 같고, 1열이 있을 때

    # 정리 *****
    # 브로드캐스트가 가능한 경우
    # (n, m) = (n, 1)
    #        = (m, ) ((1, m) => (m, )로도 쓸 수 있음)

    # 2차원 array와 1차원 array 간의 브로드캐스트
    # (n, m) array와 (m,) 1차원 array는 브로드캐스트가 가능
    # (n, m) array와 (n,) 1차원 array인 경우는,
    # 1차원 array를 (n,1) shape으로 reshape를 하면 브로드캐스트가 가능.


    # 실습 # ch1. p 35
    np.random.seed(2020)
    X = np.random.randint(10, size=(2, 3))
    print('행렬 X : \n', X)
    # 1) X 의 모든 원소들 중 최댓값을 찾아서 X 에서 찾은 최댓값(m)을 찾아서 X - m 을 계산하여 출력
    m = np.max(X)
    print('X max :', m)
    # print('m shape :', m.shape)  # scalar 는 shape 이 없다~
    print('X - m : \n', X - m)
    # 2) X 의 axis = 0 방향의 최댓값들(각 컬럼별 최댓값, 0번 축=col) 을 찾아서 그 원소가 속한 컬럼의 최댓값을 뺀 행렬을 출력
    m_by_col = np.max(X, axis=0)
    print('X max by column : ', m_by_col)
    print('X - m by column : \n', X - m_by_col)
    # 3) X 의 axis = 1 방향의 최댓값들(각 행별 최댓값, 1번 축=row) 을 찾아서 그 원소가 속한 행의 최댓값을 뺀 행렬을 출력
    m_by_row = np.max(X, axis=1)
    print('X max by row : ', m_by_row)
    biggest_by_row_for_minus = m_by_row.reshape((2, 1))
    print('X - m by row : \n', X - biggest_by_row_for_minus)

    # transpose : reshape 이 필요 없당 
    X_t = X.T  # transpose 행과 열을 바꾼 행렬
    print('X : \n', X)
    print('X_t : \n', X_t)
    m = np.max(X_t, axis=0)  # 전치 행렬에서 axis=0 방향으로 최댓값 찾음
    result = X_t - m  # 전치행렬에서 최댓값을 뺌
    result = result.T  # 전치행렬을 원래 행렬 모양으로 되돌림

    # 4. X 의 각 원소에서 그 원소가 속한 컬럼의 최댓값을 뺀 행렬의 컬럼별 원소의 합
    # transpose 방법 사용
    print('4. X : \n', X)
    m = np.max(X, axis=0)
    print('4. m : \n', m)
    print('4. X - m : \n', X - m)
    print('4. sum(X - m) : \n', np.sum(X - m, axis=0))
    # 5. X 의 각 원소에서 그 원소가 속한 행(row)의 최댓값을 뺀 행렬의 행별 원소의 합
    X_t = X.T
    print('5. X : \n', X)
    print('5. X_t : \n', X_t)
    m = np.max(X_t, axis=0)
    print('5. m : \n', m)
    result = (X_t - m)
    print('5. X - m : \n', result.T)
    # transpose 이므로~행과 열이 바뀌었다. 나중에 행별 원소의 합 하려면 shape 하면서 고치쟝
    print('5. sum(X - m) : \n', np.sum(result, axis=0))


    # 표준화 : (x - mu) / sigma
    # 표준화를 수행할 때 컬럼에 따라서 수행함 (axis = 0 방향으로 표준화)  *****
    print('X for standardization :\n', X)
    X_new = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(X_new)

    # if 표준화를 수행할 때 행(row)에 따라서 수행함 (axis = 1 방향으로 표준화) (많지 않음) -> 추천하지 않음
    print('X for standardization :\n', X)
    mu = np.mean(X, axis=1).reshape((2, 1))
    print('X mean for standardization : \n', mu)
    stddev = np.std(X, axis=1).reshape((2, 1))
    print('X std for standardization : \n', stddev)
    X_new = (X - mu) / stddev
    print(X_new)

    # Transpose 를 이용하여 행 표준화 -> 추천! 모든 계산을 transpose 하여 axis=0으로 표준화
    X_t = X.T
    X_new = (X_t - np.mean(X_t, axis=0)) / np.std(X_t, axis=0)
    X_new = X_new.T
    print(X_new)
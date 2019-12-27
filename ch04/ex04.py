"""
np.size 해석
"""

import numpy as np

a = np.array([1, 2, 3])
print('dim:', a.ndim)
print('shape:', a.shape)
print('size:', a.size)
print('length:', len(a))

A = np.array([[1, 2, 3],
              [1, 2, 3]])
print('dim:', A.ndim)
print('shape:', A.shape)
print('size:', A.size)  # 2차원 배열의 모든 원소 (각각의 원소의 갯수) 2*3=6 -> 행 * 열
print('length:', len(A))

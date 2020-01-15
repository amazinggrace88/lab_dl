import numpy as np

def cross_correlation_1d(x, w):
    """교차상관"""
    nx = len(x)
    nw = len(w)
    n = nx - nw + 1
    conv = []
    if nx >= nw:
        for i in range(n):
        x_sub = x[i:i + nw]
        fma = np.sum(x_sub * w)

    else:
        print('교차상관을 할 수 없습니다')
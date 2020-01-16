import numpy as np

from lab_dl.lab_dl.common.util import im2col

if __name__ == '__main__':
    np.random.seed(116)
    x = np.random.randint(10, size=(1, 3, 4, 4))
    col = im2col(x, 2, 2, 2, 0)
    col = col.reshape(-1, 2*2)
    print(col.shape)  # (12, 4)
    out = np.max(col, axis=1)
    out = out.reshape(1, 2, 2, 3)
    out = out.transpose(0, 3, 1, 2)
    print(out)


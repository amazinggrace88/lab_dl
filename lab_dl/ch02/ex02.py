import numpy as np
# array : vector


def and_gate(x):
    # x 는 [0,0], [0,1], [1,0], [1,1] 중 하나인 numpy.ndarray 타입
    # w = [w1, w2] 인 numpy.ndarray 가중치와 bias b를 찾음
    w = [1, 1]
    b = 1
    y = x[0] * w[0] + x[1] * w[1] + b
    if y >= 3:
        return 1
    else:
        return 0


def and_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test > 2:
        return 1
    else:
        return 0


def nand_gate(x):
    # x 는 [0,0], [0,1], [1,0], [1,1] 중 하나인 numpy.ndarray 타입
    w = [1, 1]
    b = 1
    y = x[0] * w[0] + x[1] * w[1] + b
    if y >= 3:
        return 0
    else:
        return 1


def nand_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test > 2:
        return 0
    else:
        return 1


def or_gate(x):
    # x 는 [0,0], [0,1], [1,0], [1,1] 중 하나인 numpy.ndarray 타입
    w = [1, 1]
    b = 1
    y = x[0] * w[0] + x[1] * w[1] + b
    if y >= 2:
        return 1
    else:
        return 0


def or_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test >= 2:
        return 1
    else:
        return 0


def test_perceptron(perceptron):
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            result = perceptron(x)
            print(x, '->', result)


if __name__ == '__main__':
    w = np.array([1, 1])
    print(w)

    print('\nAND:')
    test_perceptron(and_gate)
    print('\nNAND:')
    test_perceptron(nand_gate)
    print('\nOR:')
    test_perceptron(or_gate)


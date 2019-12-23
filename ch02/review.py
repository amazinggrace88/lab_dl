# what is perceptron?
import numpy as np

def and_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.dot(w) + b
    if test > 2:
        return 1
    else:
        return 0


def nand_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.bot(w) + b
    if test > 2:
        return 0
    else:
        return 1


def or_gate(x):
    w = np.array([1, 1])
    b = 1
    test = x.bot(w) + b
    if test >= 2:
        return 1
    else:
        return 0


if __name__ == '__main__':
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'{x1}, {x2}')  # 0, 1 번갈아서 나옴


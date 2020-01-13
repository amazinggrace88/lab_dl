"""
ch3 신경망

# perceptron
- 두개의 입력값 (x1, x2)
- 출력값 y
 a = x1 * w1 + x2 * w2 + b 계산 
 y = 1, a > 임계값
 y = 0, a < 임계값
- 활성화 함수(activation function)
신경망의 뉴런(neuron)에서는 입력 신호의 가중치 합을 출력값으로 변환해주는 함수가 존재
(ex_ a > 임계값)

활성화 함수의 종류
1. 계단함수
2. sigmoid 함수 - 미분 가능하다는 장점 있음
3. ReLU 함수 - 학습이 빨라지고, 연산비용 작음. 구현 매우 간단함
4. tanh 함수 - 쌍곡선 함수 중 하나, 시그모이드 함수를 transformation 해서 얻을 수 있다
"""
import numpy as np
import math
import matplotlib.pyplot as plt


# 1. 계단함수
def step_function(x):
    """
    numpy 는 element 별로 연산을 한다.
    리스트의 원소별로 >0을 계산한다.
    :param
    :return
    """
    y = x > 0
    return y.astype(np.int)
    # np.array 를 사용하는 함수에서는 for 문 사용하지 말자~ 이미 그 안에 for 문 성격이 들어있다.
    # np.array 배열에 부등호 연산을 수행하면 배열의 원소 각각에 부등호 연산을 수행한 bool 배열이 생성됨
    # type 변환 - astype 로 type 을 변환함


def step_function(x):
    result = [1 if x_i > 0 else 0
              for x_i in x]
    return result


def step_function(x):
    result = []
    for x_i in x:
        if x_i > 0:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


# 2. sigmoid 함수
def sigmoid(x):
    """sigmoid = 1 / (1 + exp(-x))"""
    """지수함수의 패키지별 비교"""
    # return 1 / (1 + math.exp(-x))
    return 1 / (1 + np.exp(-x))
    # np.array 의 특성
    # x에 여러가지 타입이 올 수 있음(Number, ndarray, Iterable(리스트나 튜플))
    # np.array 를 사용하는 함수에서는 for 문 사용하지 말자~ 이미 그 안에 for 문 성격이 들어있다.


# 3. relu 함수 (렐루)
def relu(x):
    """ReLU (Rectified Linear Unit)
    y = x, if x > 0
    y = 0, otherwise
    # 중요 ! x값을 그대로 리턴하는 선형의 성질을 지닌다
    """
    # 정류시키다 -> 전기회로에서 - 흐름을 차단한다. (x<0이하를 차단하여 0을 출력)
    return np.maximum(0, x)
    # 0, x를 비교하여 max를 찾는다.


def relu(x):
    result = []
    for x_i in x:
        if x_i > 0:
            result.append(x_i)
        else:
            result.append(0)
    return np.array(result)


def relu(x):
    return [x_i if x_i > 0 else 0 for x_i in x]


def tanh(x):
    y = 2 * x
    return 2 * sigmoid(y) - 1


if __name__ == '__main__':
    x = np.arange(-3, 4)  # like python range(start, end) 를 array 로 만들어준다.
    print('x =', x)
    # for x_i in x:
    #     print(step_function(x_i), end=' ')
    print('y = ', step_function(x))  # array 자체를 넣고 리턴값도 array 로 나오게 만들자

    # 2.
    print('sigmoid = ', sigmoid(x))  # 자체가 for 문!
    # sigmoid 를 여러개 조합하여 계산하기 때문에 ch01_review.ndarray 가 더 편하다

    # for x_i in x:
    #     print(sigmoid(x_i), end=' ')   # math.exp 는 list 를 파라미터로 넣으면 안되요~

    # 2. graph
    # step 함수, sigmoid 함수를 하나의 그래프에 그리기
    x = np.arange(-10, 10, 0.01)  # [10, 9.99, ..]
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = tanh(x)
    plt.plot(x, y1, label='Step Function')
    plt.plot(x, y2, label='Sigmoid Function')
    plt.plot(x, y3, label='Hyperbolic Tangent Function')
    plt.legend()
    plt.show()

    # 3. ReLU
    x = np.arange(-3, 4)
    relus = relu(x)  # array 자체가 파라미터로 들어간다
    print('ReLU = ', relus)
    plt.plot(x, relus)
    plt.title('ReLU')
    plt.show()

    # 4. tanh
    x = np.arange(-3, 4)
    tanh = tanh(x)
    print('Hyperbolic Tangent : ', tanh)

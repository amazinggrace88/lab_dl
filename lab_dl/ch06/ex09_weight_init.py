"""
6.2 가중치 초기값 (Weight)
- init_W 형태(ex10_twolayer)
 평균 0, 편차 1인 정규분포의 random 값 : -0.5 ~ 0.5 사이의 값이 가장 많이 뽑힌다.
 random 값을 준 이유 : 초기값을 0으로 주었다간 X dot W 이 모두 0
                    초기값을 1로 주었다간 X dot W 이 모두 같아질 것이다.
                    결론 = W 의 값들에 따라 학습이 될 수도 있고, 안 될 수도 있다.

** 신경망의 파라미터인 가중치행렬(W) 를 처음에 어떻게 초기화 하느냐에 따라 신경망의 학습 성능이 달라질 수 있다. **

weight(가중치) 의 초기값을 모두 0 또는 균일한 값으로 정하면 학습이 이루어지지 않기 때문에
weight(가중치) 의 초기값은 보통 정규 분포를 따르는 난수를 랜덤하게 추출해서 만듦.
그런데 정규분포의 표준편차에 따라 학습 성능이 달라진다. why? 표준편차가 활성화 함수에 따라 또한 변화하기 때문

1) weight 행렬의 초깃값을 N(0, 1) 분포를 따르는 난수로 생성하면
활성화 함수를 지나는 값(즉, 활성화값)들이 0 과 1 주위에 치우쳐서 분포하게 된다.
- why? 각 층의 활성화값들이 0과 1에 치우쳐 분포되므로 여기서 사용한 시그모이드 함수는 출력이 0, 1 에 가까워지면 그 미분은 0에 다가간다.
        따라서 데이터가 0과 1에 치우쳐 분포하게 되면 역전파의 기울기 값이 점점 작아지다가 사라진다.
- 역전파의 gradient 값들이 점점 작아지다가 사라지는 현상 (기울기 소실, gradient vanishing) 이 발생하게 됨

2) weight 행렬의 초깃값을 N(0, 0.01) 분포를 따르는 난수로 생성하면
활성화 값들이 0.5 주위에 치우쳐서 분포하게 된다
- why? 표준편차가 0.01 인 정규분포의 범위를 활성화 함수의 범위로 설정한다. 활성화 함수의 gradient 가 0.5 가까이 된다.
- 뉴런 1 개 짜리 신경망과 거의 비슷한 결과를 도출한다. (다수의 뉴런이 거의 같은 값을 출력하고 있으므로)
- 이러한 현상을 표현력 (representational power) 제한이라고 한다.

해결책(활성화 함수에 따라 달라진다)
1) Xvaier 초기값 : 이전 계층의 노드(뉴런)의 갯수가 n 개이면 N(0, sqrt(1/n))인 분포를 따르는 난수로 생성하는 것.
                  - 활성화 함수가 sigmoid, tanh 인 경우 사용하기 용이 (ReLU 는 부적합)
                  - 층마다 n 이 달라짐에 주의할 것 (입력값과 가중치를 dot 하기 때문에 이전 계층의 노드가 중요하다는 의견이 있음)

2) He 초기값 : 이전 계층의 노드(뉴런)의 갯수가 n 개이면 N(0, sqrt(2/n))인 분포를 따르는 난수로 생성하는 것.
             - 가중치 행렬의 초기값을 2배로 주었다.
             - 활성화 함수가 ReLU 함수인 경우 사용하기 용이
             - 층마다 n 이 달라짐에 주의할 것

"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


if __name__ == '__main__':
    # 은닉층에서 자주 사용하는 3가지 활성화 함수 그래프
    x = np.linspace(-10, 10, 1000)
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    plt.title('Activation Function')
    plt.plot(x, y_sig, label='Sigmoid')
    plt.plot(x, y_tanh, label='Hyperbolic Tangent')
    plt.plot(x, y_relu, label='ReLU')
    plt.ylim((-1.5, 1.5))
    plt.axvline(color='black')
    plt.axhline(color='black')
    plt.legend()
    plt.title('Activation Functions')
    plt.show()
    
    """
    그래프 해석
    hyperbolic tangent : -1 ~ 1 범위에서 작동, 0 이상인 양수에서 양수가 된다.
    """

    """
    Weight 를 임의의 양수, 정규분포로 만들고, 활성화 함수에 따라 값이 어떻게 변하는지 관찰
    """
    # step 1. 가상의 신경망에서 사용할 테스트 데이터를 생성
    np.random.seed(108)
    x = np.random.randn(1000, 100)  # 정규화된 mini-batch
    # step 2. 은닉층 생성
    node_num = 100  # 은닉층의 노드(뉴런) 갯수
    hidden_layer_size = 5  # 은닉층 갯수
    activations = dict()  # 데이터가 각 은닉층을 지났을 때 출력되는 값들을 저장
    weight_init_types = {
        'std=0.01': 0.01,
        'Xavier': np.sqrt(1 / node_num),
        'He': np.sqrt(2 / node_num)
    }
    input_data = np.random.randn(1_000, 100)

    for k, v in weight_init_types.items():
        x = input_data
        # 입력 데이터 x를 5개의 은닉층을 통과시킴.
        for i in range(hidden_layer_size):
            # 은닉층에서 사용하는 가중치 행렬:
            # 평균 0, 표준편차 1인 정규분포(N(0, 1))를 따르는 난수로 가중치 행렬 생성
            # w = np.random.randn(node_num, node_num)
            # w = np.random.randn(node_num, node_num) * 0.01  # N(0, 0.01)
            # w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num)  # N(0, sqrt(1/n))
            # w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)  # N(0, sqrt(2/n))
            w = np.random.randn(node_num, node_num) * v
            a = x.dot(w)  # a = x @ w
            # x = sigmoid(a)  # 활성화 함수 적용 -> 은닉층의 출력(output)
            # x = tanh(a)
            x = relu(a)
            activations[i] = x  # 그래프 그리기 위해서 출력 결과를 저장

        for i, output in activations.items():
            plt.subplot(1, len(activations), i + 1)
            # subplot(nrows, ncols, index). 인덱스는 양수(index >= 0).
            plt.title(f'{i + 1} layer')
            plt.hist(output.flatten(), bins=30, range=(-1, 1))
        plt.show()



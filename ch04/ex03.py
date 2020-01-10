"""
손실함수의 종류
- 평균 제곱 오차 (MSE) - 회귀문제(미래의 숫자(수치)를 예측)
- 교차 엔트로피 (Cross Entropy) - 실제값과 예측값의 분포를 확인

- 교차 엔트로피 (Cross Entropy)
실제값과 오차값을 교차시켜 만든다.
Entropy = - (true_value) * log (expected_value) 의 합

E = - sum_{i} (true_value_i) * log (expected_value_i)

조건 ) one_hot_encoding 이 되어있는 경우
의미 ) 데이터 1개에 대해 [1,

교차 엔트로피의 gradient 를 계산하여 + -> - 방향으로 살짝 변화시키기
교차 엔트로피의 gradient 를 계산하여 - -> + 방향으로 살짝 변화시키기

"""
import pickle
import numpy as np
from lab_dl.ch03.ex11 import forward2
from lab_dl.dataset.mnist import load_mnist


def _cross_entropy(y_pred, y_true):
    # y_pred, y_true 1차원 배열이라고 가정
    delta = 1e-7  # log0 = -inf 가 되는 것 방지하기 위해 더해줄 값
    return -np.sum(y_true * np.log(y_pred + delta))  # 값 1 개 : np.sum()
    # np.log(y_pred + delta) -> broadcasting 으로 delta 복붙해서 더하기, log 각각 씌워짐
    # (y_true * np.log(y_pred + delta)) -> 원소끼리 곱
    # row 하나씩 넣지 않아도, 즉, 10*10행렬 들어가도 됨


def cross_entropy(y_pred, y_true):
    # 1차원, 2차원일 때 모두 고려
    if y_pred.ndim == 1:  # data 갯수 1개
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true) / len(y_pred)  # .shape[0]: row 의 갯수 (2차원 배열에서의 row) = len(y_pred)
    return ce


"""
_cross_entropy 를 써서 cross_entropy 의 차원에 따라 계산 다르게 함
"""


def one_hot_encoding(y_true, y_index):
    pass

    
if __name__ == '__main__':
    (T_train, y_train), (T_test, y_test) = load_mnist(one_hot_label=True)
    y_true = y_test[:10]

    # 신경망 생성
    with open('../ch03/sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    y_pred = forward2(network, T_test[:10])

    # 교차 엔트로피 계산
    # pred 가 정답일 경우
    print('true : ', y_true[0])  # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] 이므로 엔트로피 곱하는 중에 0을 곱하여 삭제됨! 값 1개와 같음
    print('pred : ', y_pred[0])
    print('Cross Entropy : ', cross_entropy(y_pred[0], y_true[0]))  # pred 가 정답일 경우의 교차 엔트로피 0.0029
    # pred 가 오답일 경우
    print('true8 : ', y_true[8])  # 5
    print('pred8 : ', y_pred[8])  # 6이 될 확률이 가장 큼
    print('Cross Entropy : ', cross_entropy(y_pred[8], y_true[8]))  # pred 가 오답일 경우의 교차 엔트로피 4.9094

    # 요약
    # 엔트로피 = 불확실성 이므로, 오답일 때의 교차 엔트로피가 커진다.
    # 엔트로피를 줄이는 것이 목적이 된다.
    # 교차 엔트로피 모두 계산하여 교차엔트로피의 평균을 계산할 수 있다.

    print('y_true : \n', y_true[:2])
    print('y_pred : \n', y_pred[:2])
    print('Cross Entropy : ', cross_entropy(y_pred, y_true))  # 행 여러개도 cross entropy 합 가능 . 교차 엔트로피 0.5206

    # 의미 없는 계산 -> one_hot_encoding 변환이 없으면, 교차 엔트로피는 의미가 없다.
    np.random.seed(1227)
    y_true = np.random.randint(10, size=10)
    print('y_true =', y_true)
    y_pred = np.array([4, 3, 9, 7, 3, 1, 6, 6, 8, 8])  # why? 교차 엔트로피는 확률 (0~1)사이 값들이 y_true, y_pred 가 되어야 함.*****
    print('평균 CE = ', cross_entropy(y_pred, y_true))  # 평균 CE =  -100.3054227135154 -> wrong!

    # 수정한 계산
    y_true = np.random.randint(10, size=10)
    print('y_true =', y_true)  # [5 7 9 3 0 4 3 5 5 3]
    # one_hot_encoding 형태로 변환 def
    y_true_2 = np.zeros((y_true.size, 10))  # 숫자분류라서 10개 (0~9) 주었음 10*10 행렬
    print('one_hot_encoding before :\n', y_true_2)
    print()
    # 변환
    for i in range(y_true.size):  # row 갯수만큼 반복
        y_true_2[i][y_true[i]] = 1

    print('one_hot_encoding after :\n', y_true_2)


    
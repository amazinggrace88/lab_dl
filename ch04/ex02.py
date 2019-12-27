"""
ex01 과 비교

"""
import pickle
import numpy as np
from lab_dl.ch03.ex05 import forward
from lab_dl.ch03.ex11 import forward2
from lab_dl.dataset.mnist import load_mnist


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist(one_hot_label=True)  # p.112 소프트맥스 함수의 출력 = 확률이다.
    print('y[10] =', y_true[:10])  # 확률이 가장 높은 한 원소만 1, 나머지 0

    # network 연결하여 신경망 돌리기
    with open('../ch03/sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)

    y_pred = forward2(network, X_test[:10])
    # print('y_pred = ', y_pred)
    print(y_true[0])
    print(y_pred[0])  # y_true - y_pred = error

    # error
    error = y_true[0] - y_pred[0]
    print('error : ', error)
    print('error**2: ', error**2)
    print('sum error**2: ', np.sum(error**2))  # 이렇게도 오차를 구할 수 있다 (책과 다름)

    # 인덱스 8번 error
    print(y_true[8])  # 정답 5 (오차 1 from y_pred[5])
    print(y_pred[8])  # 6으로 예측 (오차 1 from y_true[5])
    print('index 8 error : ', (np.sum(y_true[8] - y_pred[8]))**2)
    # 교재에서는 ( 실제값 - 예측값 )**2 으로만으로도 인덱스 1개는 찾을 수 있다는 것이다.
    # 따라서 error 를 줄여나가면 된다.


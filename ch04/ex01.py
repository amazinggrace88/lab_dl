"""
machine learning : deep learning (기계학습의 한 종류)

machine learning
- 가중치가 알고리즘 안에서 기계가 자동으로 학습하여 정해짐
deep learning
- 신경망이라는 알고리즘을 가지고 심층 학습하는 기계학습의 한 종류
"""
import math
"""
# dataset 에 따라 지도학습, 비지도학습 나뉨
# 기계학습 : 데이터의 가중들에 임의로 숫자를 넣어보고 예측값과 비슷한지 보고, 가중치를 변경하는 과정
# 신경망   : 신경망층에서 기계학습을 수행

신경망 층들을 지나갈 때 사용되는 가중치(weight) 행렬, 편향(bias) 행렬들을 찾는 게 목적.
오차를 최소화하는 가중치 행렬들을 찾음.

손실함수(loss function) = 비용함수(cost function) = 오차(error) 최소화! 
즉, 함수의 최솟값을 찾는 문제 

손실함수의 종류
- 평균 제곱 오차 (MSE)
- 교차 엔트로피 (Cross Entropy)
"""
import numpy as np
from lab_dl.dataset.mnist import load_mnist


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_true) = load_mnist()
    print('y[10] =', y_true[:10])
    # 10 개의 테스트 데이터 이미지들의 예측값 (회귀 - 분류가 아님!)
    y_pred = np.array([7, 2, 1, 0, 4, 1, 4, 9, 6, 9])
    print('y_pred =', y_pred[:10])

    # MSE 산출 과정 : 각각의 데이터의 오차를 계산하여 제곱, 그 후 평균한다.

    # 오차
    error = y_pred - y_true[:10]
    print('error = ', error)

    # 오차 제곱(squared error)
    sq_err = error ** 2
    print('squared error =', sq_err)

    # 평균 제곱 오차
    mse = np.mean(sq_err)
    print('MSE = ', mse)

    # RMSE : Root Mean Squared Error
    print('RMSE = ', np.sqrt(mse))


    # 책 - 나혼자 해보기
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # ex1 '2'일 확률이 가장 높다고 추정함(0.6) - one_hot_encoding 로 확률 추정함 (정답)
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    mse = mean_squared_error(np.array(y), np.array(t))
    print(mse)  # 0.0975

    # ex2 '7'일 확률이 가장 높다고 추정함(0.6) - one_hot_encoding 로 확률 추정함 (오답)
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    mse = mean_squared_error(np.array(y), np.array(t))
    print(mse)  # 0.5975


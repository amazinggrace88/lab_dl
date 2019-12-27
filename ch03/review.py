"""
신경망을 행렬로 나타내기
"""
import pickle
import numpy as np

from lab_dl.dataset.mnist import load_mnist


def init_network():
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())  # keys 확인
    return network





if __name__ == '__main__':
    # data bring in !
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 초기 객체 생성
    network = init_network()

    # X_test : 2차원 배열 parameter - predict 함수
    # mini_batch = 100 설정
    mini_batch = 100

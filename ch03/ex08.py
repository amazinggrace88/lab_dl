"""
MINIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어옴
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())


if __name__ == '__main__':
    init_network()
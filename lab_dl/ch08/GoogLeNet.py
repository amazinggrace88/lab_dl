"""
GoogLeNet
p.271 그림 8-11 - concatenate
"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, concatenate

# 입력 텐서 생성
input_tensor = Input(shape=(784, ))
x1 = Dense(64, activation='relu')(input_tensor)  # input_tensor 를 함수 파라미터로 주어 연결하였다.
x2 = Dense(64, activation='relu')(input_tensor)
concat = concatenate([x1, x2])  # 두 개의 출력을 하나로 묶어주는 역할
# 연결된 텐서를 다음 계층으로 전달
x = Dense(32, activation='relu')(concat)  # concat 자체를 함수 파라미터로 주어 연결하였다.
# 출력층 생성
output_tensor = Dense(10, activation='softmax')(x)

# 모델 생성
model = Model(input_tensor, output_tensor)
model.summary()
# connected to 가 나타남
# 50240 = 784*64 + 64
# dense : input 1과 연결되어 있다.
# dense_1: input 1과 연결되어 있다.
# 128 개를 64, 64 로 나눌것 -> 왜 나누나?
# concatenate : 연결하는 역할만 / W, b 없음

"""
ResNet 
p.272 그림 8-12 - add
"""
# 기울기 소실 -> convolution 층을 지나면서 --> whu?
print()
input_tensor = Input(shape=(784,))
sc = Dense(32, activation='relu')(input_tensor)  # 기울기가 소실되지 않은
x = Dense(32, activation='relu')(sc)
x = Dense(32, activation='relu')(x)
x = Add()([x, sc])  # Add() 대문자로 시작하는 것 - 클래스/
output_tensor = Dense(10, activation='softmax')(x)

# 모델 생성
model = Model(input_tensor, output_tensor)
model.summary()
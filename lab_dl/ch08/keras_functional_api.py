"""
keras 함수형 API
"""
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input

print('기존 방식')
# layers  X(n, 64) -> Dense(32) -> Relu -> Dense(32) -> Relu ->  Dense(10) -> Softmax
seq_model = Sequential()  # 모델 정의
seq_model.add(layers.Dense(units=32, activation='relu', input_shape=(64, )))
# unit : Dense 가 가지고 있는 뉴런 갯수
# activation : 활성화 함수
# input shape : numpy shape 모양대로 신경망에 들어갈 모양을 정한다.첫번째 layer만 input shape 필요함
seq_model.add(layers.Dense(units=32, activation='relu'))
seq_model.add(layers.Dense(units=10, activation='softmax'))
# 모델은 w의 모양을 정의하는 것, 컴파일 후 학습과정(fit)에서 batch_size, epoch, .. 적용

# 모델을 요약함
seq_model.summary()
# 신경망의 param - Weight, bias
# dense_0 : 64 * 32 + 32
# dense_1 : 32 * 32 + 32
# dense_2 : 32 * 10 + 10
# 경사하강 식으로 갱신해야 할 파라미터가 총 3466 이다.

print()
print('keras 의 함수형 API 기능을 사용해서 신경망 생성하는 방법')
# input 객체 생성 -> 필요한 layer 객체를 생성 -> instance 호출 -> Model 객체(like Sequential)을 제일 마지막으로 만든다.
input_tensor = Input(shape=(64, ))  # 데이터를 주지 않고, 입력 텐서의 shape 만 결정함, 변수가 64개 있다고 알린다.

# 첫번째 은닉층(hidden layer) 생성 & 인스턴스 호출을 사용해서 입력 데이터 전달
x = layers.Dense(32, activation='relu')(input_tensor)
# Dense 라는 인스턴스에 함수처럼 파라미터 주어 인스턴스 호출, input 층과 dense 층을 연결

print('input_tensor type : ', type(input_tensor))  # tensor
print('x type : ', type(x))  # tensor

# 두번째 은닉층(hidden layer) 생성 & 첫번째 은닉층의 출력을 입력으로 전달
x = layers.Dense(32, activation='relu')(x)

# 출력층 생성 & 두번째 은닉층의 출력을 입력으로 전달
output_tensor = layers.Dense(10, activation='softmax')(x)

# 모델 객체 생성
model = Model(input_tensor, output_tensor)
# input, output 을 연결하여 model 을 준다. --> 중간단계는 모두 연결되었기 때문.

# 모델을 요약함
model.summary()
# input 쪽에는 신경망 param이 없다.


# 두 모델 모두 모델 생성 후 -> 모델 컴파일(compile) -> 모델 학습(fit) -> 모델 평가/예측 하는데 사용(evaluate method, predict method)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # one-hot-encoding 하지 않았을 때
x_train = np.random.random(size=(1000, 64))
y_train = np.random.randint(10, size=(1000, 10))  # one-hot-encoding 처럼 만들기 위해서 (1000, 10)10개 줌.
model.fit(x_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(x_train, y_train)

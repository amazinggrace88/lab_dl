"""
SimpleConvNet (간단한 CNN)을 활용한 MNIST 손글씨 이미지 데이터 분류
"""
from matplotlib import pyplot as plt
from lab_dl.lab_dl.ch07.simple_convnet import SimpleConvNet
from lab_dl.lab_dl.common.trainer import Trainer
from lab_dl.lab_dl.dataset.mnist import load_mnist
import numpy as np


# 데이터 로드
(X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=True, flatten=False)  # 신경망에 쓸 때는 정규화

# # 테스트 시간을 줄이기 위해서 데이터 사이즈를 줄였다 (원래는 데이터 사이즈 줄이면 안됨)
# X_train, Y_train = X_train[:5000], Y_train[:5000]
# X_test, Y_test = X_test[:1000], Y_test[:1000]

# CNN 생성
cnn = SimpleConvNet()

# 테스트 도우미 클래스
trainer = Trainer(network=cnn,  # 사용할 네트워크 이름
                  x_train=X_train,
                  t_train=Y_train,
                  x_test=X_test,
                  t_test=Y_test,
                  epochs=20,  # epoch 갯수
                  mini_batch_size=100,  # 미니배치 사이즈
                  optimizer='Adam',  # 옵티마이저
                  optimizer_param={'lr': 0.01},  # parameter
                  evaluate_sample_num_per_epoch=1000)
# evaluate_sample_num_per_epoch=1000 : 저자가 생각하는 에포크는 원하는 만큼 미니배치 갯수가 몇개든 지정한 갯수를 돌리면 1 epoch 라고 준다.

# 테스트 실행
trainer.train()

# 학습이 끝난 후 파라미터들을 파일에 저장
cnn.save_params(file_name='cnn_params.pkl')

# 그래프(x축 - epoch, y축 - 정확도)
x = np.arange(20)  # epoch 숫자와 맞춰준다
plt.plot(x, trainer.train_acc_list, label='train accuracy')
plt.plot(x, trainer.test_acc_list, label='test accuracy')  # 과적합을 보기 위해 train/test 같은 공간에 그려준다.
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

